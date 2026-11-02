from pathlib import Path
import random
from typing import Dict, Any, Tuple, List

import cv2
import numpy as np

from real_vas.utils.camera_motion_tools.components import PRESETS
from real_vas.utils.camera_motion_tools.utils import create_camera_motion, save_outputs
from real_vas.utils.occlusion_utils import get_occlusion_info, OcclusionLevel


def draw_view_on_frame(original_frame: np.ndarray, new_frame_shape: tuple, matrix: np.ndarray) -> np.ndarray:
    """
    Draws the bounding box of the transformed view on the original frame
    using a 3x3 perspective transformation matrix.

    Args:
        original_frame (np.ndarray): The original, untransformed frame.
        new_frame_shape (tuple): The shape (H, W) of the new (output) frame.
        matrix (np.ndarray): The 3x3 perspective transformation matrix.

    Returns:
        np.ndarray: A copy of the original frame with the view polygon drawn on it.
    """
    if matrix.shape != (3, 3):
        raise ValueError("This function requires a 3x3 transformation matrix.")

    h, w = new_frame_shape[:2]

    # Corners of the new frame. Note the shape (1, 4, 2) is required by cv2.perspectiveTransform
    corners_new = np.array([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]
    ], dtype=np.float32).reshape(1, 4, 2)

    # Invert the 3x3 matrix to map from the new frame back to the original
    try:
        M_inv = np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        print("Warning: Could not invert the transformation matrix (it's singular). Skipping drawing.")
        return original_frame.copy()

    # Use perspectiveTransform to find the corner locations in the original frame
    corners_original = cv2.perspectiveTransform(corners_new, M_inv)

    frame_with_view = original_frame.copy()

    # Draw the resulting polygon
    cv2.polylines(
        img=frame_with_view,
        pts=[np.int32(corners_original)],
        isClosed=True,
        color=(0, 255, 0),  # Bright green
        thickness=2,
        lineType=cv2.LINE_AA
    )

    return frame_with_view


class CameraMotionApplier:
    """
    A callable class to apply camera motion presets to a sequence of frames and masks.

    This class iterates through a set of predefined camera motion presets,
    evaluates their impact on background visibility, and selects an optimal
    preset that maintains a high level of background visibility. The results
    are then saved to a specified output directory.
    """

    def __init__(self, output_dir: str | Path | None = None, visibility_threshold: float = 0.95,
                 presets_to_run: List | None = None):
        """
        Initializes the CameraMotionApplier.

        Args:
            output_dir (str | Path | None): The root directory where results will be saved.
            visibility_threshold (float): The minimum average background visibility score
                                          for a preset to be considered 'good'. Defaults to 0.95.
            presets_to_run: List | None: A list of preset names to run. If None, all presets are used.
        """
        if output_dir is not None:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(exist_ok=True, parents=True)
        self.visibility_threshold = visibility_threshold
        # A dictionary of camera motion presets
        self.presets: Dict[str, Any] = PRESETS if presets_to_run is None else {
            k: v for k, v in PRESETS.items() if k in presets_to_run}

    def __call__(self, frames: np.ndarray, fg_masks: np.ndarray, bg_masks: np.ndarray, video_name: str,
                 save_output: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        """
        Applies camera motion to the given frames and masks and saves the results.

        Args:
            frames (np.ndarray): A numpy array of frames shaped (T, H, W, C).
            fg_masks (np.ndarray): A numpy array of foreground masks shaped (T, H, W).
            bg_masks (np.ndarray): A numpy array of background masks shaped (T, H, W).
            video_name (str): The name of the video, used for creating output subdirectories.
            save_output (bool): Whether to save the output frames and masks. Defaults to False.

        Returns:
            A tuple containing the chosen frames, foreground masks, background masks,
            and the name of the chosen preset.
        """
        preset_results = {}

        # 1. Evaluate all presets
        for preset_name in self.presets.keys():
            new_frames, new_fg_masks, new_bg_masks, bg_vis_scores, _ = create_camera_motion(
                frames, fg_masks, bg_masks, selected_presets=preset_name
            )

            # The first frame must have a fully visible background to be a valid candidate
            if bg_vis_scores[0] != 1.0:
                continue

            occl_infos = get_occlusion_info(new_bg_masks, all_fg_alphas=new_fg_masks > 0.8)
            any_none_occlusion = np.array([o.level == OcclusionLevel.NONE for o in occl_infos])
            if any_none_occlusion.any():
                # If any frame has NONE occlusion, skip this preset as it means it has left the frame
                continue

            # If a background point was not visible in the original mask, it's okay if it's
            # not visible now. This prevents penalizing the score for inherently occluded points.
            bg_vis_scores[bg_masks.sum(axis=(1, 2)) == 0] = 1.0

            preset_results[preset_name] = {
                "point_still_vis": bg_vis_scores.mean(),
                "new_frames": new_frames,
                "new_fg_masks": new_fg_masks,
                "new_bg_masks": new_bg_masks,
            }

        # 2. Filter for presets that meet the visibility threshold
        good_presets = {
            key: value
            for key, value in preset_results.items()
            if value['point_still_vis'] > self.visibility_threshold
        }

        # 3. Choose the best preset
        preset_keys = list(good_presets.keys())
        if preset_keys:
            # Randomly select one of the good presets
            chosen_preset_name = random.choice(preset_keys)
            chosen_output = preset_results[chosen_preset_name]
            chosen_frames = chosen_output['new_frames']
            chosen_fg_masks = chosen_output['new_fg_masks']
            chosen_bg_masks = chosen_output['new_bg_masks']
        else:
            # If no preset is good enough, default to a static camera (no motion)
            chosen_preset_name = "static"
            chosen_frames = frames
            chosen_fg_masks = fg_masks
            chosen_bg_masks = bg_masks

        print(f"Chosen preset for '{video_name}': '{chosen_preset_name}'")

        if save_output:
            # 4. Save the final chosen output
            video_out_dir = self.output_dir / video_name
            video_out_dir.mkdir(exist_ok=True)
            final_out_dir = video_out_dir / chosen_preset_name
            final_out_dir.mkdir(exist_ok=True)
            save_outputs(final_out_dir, chosen_frames, chosen_fg_masks, chosen_bg_masks)

        return chosen_frames, chosen_fg_masks, chosen_bg_masks, chosen_preset_name
