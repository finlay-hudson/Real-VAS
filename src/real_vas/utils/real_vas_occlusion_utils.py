from typing import Optional

import cv2
import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation, median_filter
from shapely.geometry import Polygon
import torch

from real_vas.utils.mask_utils import get_bbox_from_binary_mask


def calculate_temporal_iou(range1: tuple[int, int], range2: tuple[int, int]) -> float:
    """Calculates the Intersection over Union for two 1D integer ranges."""
    start1, end1 = min(range1), max(range1)
    start2, end2 = min(range2), max(range2)

    intersection_start = max(start1, start2)
    intersection_end = min(end1, end2)

    intersection = max(0, intersection_end - intersection_start + 1)
    if intersection == 0:
        return 0.0

    len1 = end1 - start1 + 1
    len2 = end2 - start2 + 1
    union = len1 + len2 - intersection

    return intersection / union


def get_overlapping_ranges(
        n1: int,
        n2: int,
        best_shift: int
) -> dict:
    """
    Calculates the overlapping frame ranges given the video lengths and the best shift.

    Args:
        n1 (int): Number of frames in video1.
        n2 (int): Number of frames in video2.
        best_shift (int): The optimal temporal shift for video2 relative to video1.
                          A positive value means video2 starts after video1.

    Returns:
        A dictionary with the start and end indices (inclusive) for the
        overlapping segment of each video.
        e.g., {'masks1_range': (start1, end1), 'masks2_range': (start2, end2)}
    """
    s = best_shift

    # Determine the absolute frame indices for the overlap period on a shared timeline
    # The first video (video1) is at timeline [0, n1-1]
    # The second video (video2) is at timeline [s, s + n2 - 1]
    overlap_start_abs = max(0, s)
    overlap_end_abs = min(n1 - 1, s + n2 - 1)

    # Check if there is any actual overlap
    if overlap_start_abs > overlap_end_abs:
        return {
            'masks1_range': None,
            'masks2_range': None,
        }

    # The range for video1 corresponds directly to the absolute overlap
    masks1_range = (overlap_start_abs, overlap_end_abs + 1)  # +1 for inclusive end index

    # The range for video2 is the absolute overlap shifted back by `s`
    masks2_start_rel = overlap_start_abs - s
    masks2_end_rel = overlap_end_abs - s
    masks2_range = (masks2_start_rel, masks2_end_rel + 1)  # +1 for inclusive end index

    return {
        'masks1_range': masks1_range,
        'masks2_range': masks2_range,
    }


def get_all_overlap_scores(video1: torch.Tensor, video2: torch.Tensor) -> torch.Tensor:
    """Helper to run the convolution and return all scores."""
    n1, h, w = video1.shape
    n2 = video2.shape[0]

    v1_flat = video1.float().view(n1, h * w).T.unsqueeze(0)
    v2_flat = video2.float().view(n2, h * w).T.unsqueeze(1)

    correlation = torch.nn.functional.conv1d(
        v1_flat, v2_flat, padding=(n2 - 1), groups=(h * w)
    )
    return correlation.sum(dim=1).squeeze(0)


def find_longest_true_run(bool_tensor: torch.Tensor) -> tuple[int, int] | tuple[None, None]:
    """Finds the start and end indices of the longest continuous run of True."""
    # Pad with False at both ends to handle runs at the start/end
    padded_tensor = torch.cat([
        torch.tensor([False]), bool_tensor.cpu(), torch.tensor([False])
    ])

    # Find where runs of True start and end
    diffs = torch.diff(padded_tensor.int())
    starts = (diffs == 1).nonzero().flatten()
    ends = (diffs == -1).nonzero().flatten() - 1

    if len(starts) == 0:
        return None, None  # No runs of True found

    # Calculate lengths and find the longest run
    lengths = ends - starts + 1
    max_idx = torch.argmax(lengths)

    return starts[max_idx].item(), ends[max_idx].item()


def trim_sequence_before_duplication(fg_data: dict, bg_data: dict, all_masks: list) -> tuple[int, int] | tuple[
    None, None]:
    """
    Checks if the FG object appears in the BG's timeframe and trims the sequence
    to the frames before that duplication occurs.
    """
    # 1. Get the original full mask of the foreground object
    original_fg_mask = all_masks[fg_data["idx"]]

    # 2. Get the slice of the FG mask during the BG's time
    fg_mask_during_bg_time = original_fg_mask[bg_data["frame_range"]]
    is_fg_present_per_frame = fg_mask_during_bg_time.any(dim=(1, 2))

    # 4. Find where duplication is ABSENT
    # A valid frame is one where the foreground is NOT present (False).
    is_duplication_absent = ~is_fg_present_per_frame

    # 5. Find the longest continuous run of valid (duplication-absent) frames
    # We can reuse our existing helper function for this!
    start, end = find_longest_true_run(is_duplication_absent)

    if start is None:
        # No period without duplication was found
        return None, None

    return start, end


def how_much_bg_occluded(fg_masks_seq, bg_masks_seq):
    masks_fg = fg_masks_seq.to(bool)
    masks_bg = bg_masks_seq.to(bool)

    # Intersection: shape (N, H, W)
    # intersection = np.logical_and(masks_bg, masks_fg)
    intersection = masks_bg & masks_fg

    # Sum pixels per mask in batch
    occluded_pixels = intersection.reshape(intersection.shape[0], -1).sum(1)
    total_bg_pixels = masks_bg.reshape(masks_bg.shape[0], -1).sum(1)

    # Avoid division by zero by setting zero denominators to 1 temporarily
    safe_total = torch.where(total_bg_pixels == 0, 1, total_bg_pixels)

    ratios = occluded_pixels / safe_total

    # If total_a_pixels was zero, ratio should be 0
    ratios = torch.where(total_bg_pixels == 0, 0, ratios)

    return ratios


def assign_fg_bg(data_1: dict, data_2: dict) -> tuple[dict, dict] | tuple[None, None]:
    """
    Determines which object is in the foreground based on median depth.

    Args:
        data_1 (dict): A dictionary containing masks, depths, etc., for the first object.
        data_2 (dict): A dictionary for the second object.

    Returns:
        A tuple containing (fg_data, bg_data), or (None, None) if a valid
        depth comparison cannot be made (e.g., due to empty masks).
    """
    # Extract the masks and depths for comparison
    masks1, depths1 = data_1["masks"], data_1["depths"]
    masks2, depths2 = data_2["masks"], data_2["depths"]

    if not (masks1.any() and masks2.any()):
        return None, None

    # Use median depth to determine which is in front
    median_depth_1 = torch.median(depths1[masks1])
    median_depth_2 = torch.median(depths2[masks2])

    # The object with the smaller median depth value is in front
    is_1_in_front = median_depth_1 < median_depth_2

    # Return the data bundles in the correct (foreground, background) order
    return (data_2, data_1) if is_1_in_front else (data_1, data_2)


def get_start_frame(masks):
    return torch.where(masks.view(masks.shape[0], -1).any(-1))[0].min()


def is_masks_occluded_by_other_objects(masks, frame_depths):
    is_mask_occluded = torch.zeros(len(masks), dtype=torch.bool, device=masks.device)
    start_frame = get_start_frame(masks)
    _is_mask_occluded, _ = find_if_any_monodepth_occlusion(masks[start_frame:], frame_depths[start_frame:])
    _is_mask_occluded = torch.from_numpy(np.array(list(_is_mask_occluded.values()))).to(masks.device)
    is_mask_occluded[start_frame:] = _is_mask_occluded

    return is_mask_occluded


def find_if_fg_obj_occluded_by_bg_obj(fg_data, bg_data, check_mask, min_thresh=0.05, epsilon=0.2):
    fg_depth_for_fg_mask = fg_data["depths"].clone()
    fg_depth_for_fg_mask[check_mask == 0] = 0
    bg_depth_for_fg_mask = bg_data["depths"].clone()
    bg_depth_for_fg_mask[check_mask == 0] = 0

    # TODO this is not robust enough between 00004 and 00009 - need to think of a better way of testing this
    return ((fg_depth_for_fg_mask < (bg_depth_for_fg_mask - epsilon)).sum([1, 2]) / check_mask.sum([1, 2])) > min_thresh


def is_on_top_of_each_other(mask_seq_a, depths_a, mask_seq_b, depths_b, ontop_thresh=0.1):
    # We dont want objects taking up exactly the same pixels
    depths_a = depths_a.clone()
    depths_b = depths_b.clone()
    depths_a[~(mask_seq_a & mask_seq_b)] = torch.nan
    depths_b[~(mask_seq_a & mask_seq_b)] = torch.nan

    # depth_diffs = torch.abs(depths_a.nanmean(dim=(1, 2)) - depths_b.nanmean(dim=(1, 2)))
    # is_on_top = depth_diffs < ontop_thresh

    depth_diffs = torch.abs(depths_a - depths_b)
    is_on_top = (depth_diffs < ontop_thresh).sum(dim=(1, 2))

    return is_on_top > 5, depth_diffs  # just if its a bit of noise


def validate_and_trim_sequence(
        fg_data: dict,
        bg_data: dict,
        masks1_already_occluded: torch.Tensor,
        masks2_already_occluded: torch.Tensor
) -> tuple[dict, dict] | None:
    """
    Validates a candidate based on two rules and trims it to the longest valid segment.
    1. The background mask must always be present.
    2. The foreground mask must not have been externally occluded.
    """
    # Rule 1: Find where the background mask is present in the sequence
    is_bg_present = bg_data["masks"].any(dim=(1, 2))

    # Rule 2: Find where the foreground mask was NOT externally occluded
    # First, get the correct original occlusion tensor for the FG object
    occlusion_tensors = [masks1_already_occluded, masks2_already_occluded]
    fg_occlusion_full = occlusion_tensors[fg_data["idx"]]

    # Get the slice of this tensor corresponding to the FG sequence's original frame numbers
    fg_occlusion_slice = fg_occlusion_full[fg_data["frame_range"]]

    # A frame is valid if the foreground was not occluded
    is_fg_valid = ~fg_occlusion_slice

    # --- Combine Rules ---
    # A frame is only valid if BOTH conditions are met.
    # Note: is_bg_present and is_fg_valid must have the same length.
    # The length of the aligned sequence is determined by the overlap, so they will match.
    is_frame_valid = is_bg_present & is_fg_valid

    # Find the longest continuous run of frames that satisfy both rules
    start, end = find_longest_true_run(is_frame_valid)

    return start, end


def find_if_any_monodepth_occlusion(manual_masks: np.ndarray | list, monodepth_results: list,
                                    md_occl_thresh: float = 0.01) -> tuple[dict, dict]:
    is_item_occluded = {}
    amounts_of_occlusion = {}
    prev_occl_pred = False
    for annotated_frame_id in range(0, len(manual_masks)):
        manual_mask = manual_masks[annotated_frame_id]
        manual_mask = manual_mask.clip(0, 1)
        occlusion_pred = True
        if manual_mask.sum() == 0:
            amount_of_occlusion = 1e9
        else:
            occlusion_pred, amount_of_occlusion = assert_if_something_in_front(manual_mask,
                                                                               monodepth_results[annotated_frame_id],
                                                                               thresh=md_occl_thresh, radius=5,
                                                                               prev_occl=prev_occl_pred)
            prev_occl_pred = occlusion_pred
        is_item_occluded[annotated_frame_id] = occlusion_pred
        amounts_of_occlusion[annotated_frame_id] = amount_of_occlusion

    return is_item_occluded, amounts_of_occlusion


def assert_if_something_in_front(img_mask: np.ndarray, frame_monodepth_results: dict,
                                 ref_mask: Optional[np.ndarray] = None, thresh: float = 0.01, radius: int = 5,
                                 min_dist: int = 3, use_bbox: bool = False, prev_occl: bool = False) -> tuple[
    bool, float]:
    # depth_img_arr = np.array(frame_monodepth_results["depth"])
    depth_img_arr = frame_monodepth_results.clone().cpu().numpy()
    img_mask = img_mask.cpu().numpy()
    depth_img_arr[img_mask == 1] = 0
    if use_bbox:
        mask_bbox = get_bbox_from_binary_mask(img_mask)
        depth_img_arr = depth_img_arr[mask_bbox[0]:mask_bbox[2], mask_bbox[1]:mask_bbox[3]]
    else:
        exp_mask = cv2.dilate(img_mask.astype(np.uint8).copy(), np.ones((2 * radius + 1, 2 * radius + 1), np.uint8))
        contours, _ = cv2.findContours(exp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # Put them in skimage format
        contours = [contour[:, 0, ::-1] for contour in contours]
        contours = filter_small_contours(contours, min_area=250)
        if len(contours) > 1:
            return True, 1e9  # If there are multiple contours, we assume occlusion
            # result_image = connect_contours(img_mask, contours)
            # exp_mask = (exp_mask | result_image)
        # exp_mask = cv2.dilate(exp_mask, np.ones((2 * radius + 1, 2 * radius + 1), np.uint8))
        depth_img_arr[exp_mask == 0] = 0

    if ref_mask is not None:
        depth_img_arr[ref_mask == 0] = 0

    # depth_map_of_mask = np.array(frame_monodepth_results["depth"])
    depth_map_of_mask = frame_monodepth_results.clone().cpu().numpy()
    # Clean up any outliers in the depth map
    depth_map_of_mask = median_filter(depth_map_of_mask, size=3)

    shrunk_mask = binary_erosion(img_mask, structure=np.ones((2, 2), dtype=np.uint8), iterations=1)
    shrunk_outline = cv2.Canny((shrunk_mask * 255).astype(np.uint8), 100, 200)
    shunk_depth_map_of_mask = depth_map_of_mask.copy()
    shunk_depth_map_of_mask[shrunk_outline == 0] = 0

    expanded_mask = binary_dilation(img_mask, structure=np.ones((3, 3), dtype=np.uint8), iterations=1)
    expanded_outline = cv2.Canny((expanded_mask * 255).astype(np.uint8), 100, 200)
    expanded_depth_map_of_mask = depth_map_of_mask.copy()
    expanded_depth_map_of_mask[expanded_outline == 0] = 0

    if ((shunk_depth_map_of_mask > 0).sum() < 20) and prev_occl:
        # The mask must be so small that it is not possible to find a nearest pixel so we assume occluded
        return True, 1e9

    # if expanded_depth_map_of_mask.sum() == 0:
    #     return True, 1e9

    result = find_nearest_pixels(shunk_depth_map_of_mask, expanded_depth_map_of_mask, min_dist=min_dist)
    occl_val = np.sum(result) / (shunk_depth_map_of_mask > 0).sum()

    return occl_val > thresh, occl_val


def is_item_leaving_frame(all_masks, frame_idx, prev_out_of_frame=False, edge_of_scene_thresh=10, simple=False):
    if all_masks[frame_idx] is None or all_masks[frame_idx].sum() == 0:
        # We just have to trust the previous out of frame state
        return prev_out_of_frame
    mask_shape = all_masks[frame_idx].shape
    if isinstance(all_masks, torch.Tensor):
        all_masks = all_masks.cpu().numpy()
    curr_bbox = get_bbox_from_binary_mask(all_masks[frame_idx])
    if (curr_bbox[0] <= 0 or curr_bbox[1] <= 0 or curr_bbox[2] >= (mask_shape[0] - edge_of_scene_thresh) or
            curr_bbox[3] >= (mask_shape[1] - edge_of_scene_thresh)):
        if prev_out_of_frame or simple:
            return True
        if (frame_idx + 1) < len(all_masks):
            next_mask = all_masks[frame_idx + 1]
            if next_mask is None or next_mask.sum() == 0:
                # Have to assume it has gone out of the frame
                return True
            next_bbox = get_bbox_from_binary_mask(all_masks[frame_idx + 1])
            if (next_bbox[0] <= 0 or next_bbox[1] <= 0 or
                    next_bbox[2] >= (mask_shape[0] - edge_of_scene_thresh) or
                    next_bbox[3] >= (mask_shape[1] - edge_of_scene_thresh)):
                return True
            # Might have just touched the edge of the frame for one frame so just assume not

    return False


def contour_area(contour):
    """ Calculate the area of a contour using its convex hull. """
    if len(contour) < 3:
        return 0.0
    polygon = Polygon(contour)
    return polygon.area


def filter_small_contours(contours: list, min_area: int = 5) -> list:
    """ Filter out contours with an area smaller than min_area. """
    return [contour for contour in contours if contour_area(contour) > min_area]


def find_best_occlusion_clump(frames: torch.Tensor, occluded_ratios: torch.Tensor):
    """
    Finds the contiguous clump of frames with the highest peak occlusion,
    ensuring the first frame of the clump is unoccluded.

    Args:
        frames: A 1D tensor of frame numbers, which may have discontinuities.
        occluded_ratios: A 1D tensor of occlusion ratios corresponding to each frame.

    Returns:
        A tensor containing the frame numbers of the best clump, or None if no
        valid clump is found.
    """
    if frames.numel() == 0:
        return None

    # Convert to NumPy for easier processing
    frames_np = frames.numpy()
    ratios_np = occluded_ratios.numpy()

    # --- Step 1: Find all contiguous clumps of frame INDICES ---
    # A split occurs where the frame number does not increment by 1
    splits = np.where(np.diff(frames_np) != 1)[0] + 1
    # np.split uses these split points to create the clumps of indices
    clumps_by_indices = np.split(np.arange(len(frames_np)), splits)

    # --- Step 2: Filter for clumps that start with zero occlusion ---
    valid_clumps = []
    for clump in clumps_by_indices:
        # Get the index of the first frame in the clump
        start_index = clump[0]
        # Check if the occlusion at that start index is 0
        if ratios_np[start_index] == 0:
            valid_clumps.append(clump)

    if not valid_clumps:
        return None

    # --- Step 3: Find the best clump among the valid ones ---
    best_clump = None
    max_peak_occlusion = -1.0

    for clump in valid_clumps:
        # Get all occlusion ratios for the current clump
        ratios_in_clump = ratios_np[clump]
        peak_occlusion_in_clump = np.max(ratios_in_clump)

        if peak_occlusion_in_clump <= 0.0:
            continue

        if peak_occlusion_in_clump > max_peak_occlusion:
            max_peak_occlusion = peak_occlusion_in_clump
            best_clump = clump

    return best_clump


def find_nearest_pixels(shrunk_mask: np.ndarray, expanded_mask: np.ndarray, min_dist: int = 3) -> np.ndarray:
    # Find the coordinates of non-zero pixels (True pixels) in the expanded mask
    expanded_coords = np.argwhere(expanded_mask)

    # Initialize an array to store the nearest values in the expanded mask
    in_front_vals = np.zeros_like(shrunk_mask, dtype=np.float32)

    for (i, j) in np.argwhere(shrunk_mask):
        # Calculate distances from the current pixel to all non-zero pixels in the expanded mask
        distances = np.sqrt((expanded_coords[:, 0] - i) ** 2 + (expanded_coords[:, 1] - j) ** 2)

        # Get the index of the closest expanded pixel
        nearest_index = np.argmin(distances)
        in_front_vals[i, j] = expanded_mask[
                                  expanded_coords[nearest_index][0], expanded_coords[nearest_index][1]] > (
                                      shrunk_mask[i, j] + min_dist)

    return in_front_vals
