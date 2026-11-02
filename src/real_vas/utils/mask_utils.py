import base64
import hashlib

import numpy as np
from pycocotools import mask as mask_utils
import torch


def convert_masks_into_rle(masks, use_base64=False):
    rle_masks = []
    for i in range(masks.shape[0]):
        m = np.asfortranarray(masks[i].astype(np.uint8))
        rle = mask_utils.encode(m)

        # If requested, encode the 'counts' bytes to a Base64 string
        if use_base64:
            rle['counts'] = base64.b64encode(rle['counts']).decode('ascii')

        rle_masks.append(rle)

    return rle_masks


def convert_rles_to_masks(rles, to_torch=True, use_base64=False):
    masks = []
    for i in range(len(rles)):
        rle_to_decode = rles[i]

        # If requested, decode the 'counts' string back into bytes
        if use_base64:
            # Create a copy to avoid modifying the original input list
            rle_to_decode = rles[i].copy()
            encoded_counts = rles[i]['counts']
            rle_to_decode['counts'] = base64.b64decode(encoded_counts)

        mask = mask_utils.decode(rle_to_decode)
        if to_torch:
            mask = torch.from_numpy(mask)
        masks.append(mask)

    if to_torch:
        return torch.stack(masks, dim=0)
    return masks


def get_bbox_corners_from_masks(masks):
    B, N, H, W = masks.shape
    device = masks.device

    # Prepare coordinate grid (H*W,)
    yy = torch.arange(H, device=device).repeat(W)
    xx = torch.arange(W, device=device).repeat_interleave(H)

    # (H*W,) → (1, H*W)
    xx = xx.unsqueeze(0)
    yy = yy.unsqueeze(0)

    all_boxes = []

    for b in range(B):
        m = masks[b].reshape(N, H * W)

        m_inv = ~m

        x_masked_min = torch.where(m_inv, W, xx)
        y_masked_min = torch.where(m_inv, H, yy)

        x_masked_max = torch.where(m_inv, 0, xx)
        y_masked_max = torch.where(m_inv, 0, yy)

        x_min = x_masked_min.amin(dim=1)
        y_min = y_masked_min.amin(dim=1)
        x_max = x_masked_max.amax(dim=1)
        y_max = y_masked_max.amax(dim=1)

        corners = torch.stack([
            torch.stack([x_min, y_min], dim=1),  # top-left
            torch.stack([x_max, y_min], dim=1),  # top-right
            torch.stack([x_max, y_max], dim=1),  # bottom-right
            torch.stack([x_min, y_max], dim=1),  # bottom-left
        ], dim=1)

        all_boxes.append(corners)

    return torch.stack(all_boxes, dim=0)


def pairwise_mask_intersection(preds: torch.Tensor, targets: torch.Tensor, chunk_size=-1) -> torch.Tensor:
    """
    Computes pairwise intersection (pixel count) between each pred and each target mask.

    Args:
        preds (torch.Tensor): (N, 1 H, W) binary masks
        targets (torch.Tensor): (M, 1 H, W) binary masks

    Returns:
        torch.Tensor: Intersection matrix of shape (N, M), where entry (i, j) is the
                      number of overlapping True pixels between preds[i] and targets[j].
    """
    assert preds.shape[1] == 1 and targets.shape[1] == 1, "Can only do one single comparisons"
    targets = targets.permute(1, 0, 2, 3)  # (1, M, H, W)
    if chunk_size < 0:
        intersection = (preds & targets).sum(dim=(2, 3))
        pred_area = preds.sum(dim=(2, 3)).clamp(min=1)  # (N, 1)
        return intersection.float() / pred_area.float()  # (N, M)

    N = preds.shape[0]
    M = targets.shape[1]

    intersections = []

    for i in range(0, N, chunk_size):
        chunk_preds = preds[i:i + chunk_size]  # (chunk_N, 1, H, W)
        row_chunks = []

        for j in range(0, M, chunk_size):
            chunk_targets = targets[:, j:j + chunk_size]  # (1, chunk_M, H, W)

            # Broadcast: (chunk_N, 1, H, W) & (1, chunk_M, H, W) -> (chunk_N, chunk_M, H, W)
            intersection_chunk = (chunk_preds & chunk_targets).sum(dim=(2, 3))  # (chunk_N, chunk_M)
            row_chunks.append(intersection_chunk)

        # Concatenate all column chunks horizontally
        row = torch.cat(row_chunks, dim=1)  # (chunk_N, M)
        intersections.append(row)

    # Concatenate all row chunks vertically
    intersections = torch.cat(intersections, dim=0)  # (N, M)
    pred_area = preds.sum(dim=(2, 3)).clamp(min=1)  # (N, 1)
    return intersections.float() / pred_area.float()  # (N, M)


def pairwise_batch_iou(preds, targets, eps=1e-6):
    """
    Computes pairwise IoU between each pred and each target.

    Args:
        preds (torch.Tensor): (N, H, W) binary masks
        targets (torch.Tensor): (M, H, W) binary masks

    Returns:
        torch.Tensor: IoU matrix of shape (N, M)
    """
    N, H, W = preds.shape
    M = targets.shape[0]

    preds = preds.view(N, 1, H, W)  # (N, 1, H, W)
    targets = targets.view(1, M, H, W)  # (1, M, H, W)

    intersection = (preds & targets).float().sum(dim=(2, 3))  # (N, M)
    union = (preds | targets).float().sum(dim=(2, 3))  # (N, M)

    iou = (intersection + eps) / (union + eps)
    return iou


def get_bbox_from_binary_mask(binary_mask):
    # Find the non-zero coordinates
    non_zero_coords = np.argwhere(binary_mask)

    # If there are no non-zero pixels, return None
    if non_zero_coords.size == 0:
        return None

    # Get the bounding box coordinates
    min_row, min_col = np.min(non_zero_coords, axis=0)
    max_row, max_col = np.max(non_zero_coords, axis=0)

    # Return the bounding box as (min_row, min_col, max_row, max_col)
    return np.array([min_row, min_col, max_row, max_col]).astype(int)


def get_mask_hash_id(mask: np.ndarray, frame_number: int = 0, hash_type='sha1', base64_output=True, length=None):
    assert mask.dtype == np.bool_ or np.issubdtype(mask.dtype, np.integer)

    mask_bytes = mask.astype(np.uint8).tobytes()

    # Include frame number as 8-byte little-endian integer
    frame_bytes = frame_number.to_bytes(8, byteorder='little', signed=False)

    combined = mask_bytes + frame_bytes

    if hash_type == 'sha1':
        digest = hashlib.sha1(combined).digest()
    elif hash_type == 'sha256':
        digest = hashlib.sha256(combined).digest()
    elif hash_type == 'md5':
        digest = hashlib.md5(combined).digest()
    else:
        raise ValueError("Unsupported hash type")

    # Convert to base64 or hex
    if base64_output:
        encoded = base64.urlsafe_b64encode(digest).decode('utf-8')
    else:
        encoded = digest.hex()

    return encoded[:length] if length else encoded


def erode_batch(masks, kernel_size=3, iterations=1):
    """
    Erode a batch of binary masks using 2D max pooling.

    Args:
        masks: Tensor of shape (B, 1, H, W) or (B, H, W)
        kernel_size: Size of the structuring element
        iterations: Number of erosion iterations

    Returns:
        Tensor of shape (B, 1, H, W)
    """
    if masks.ndim == 3:  # Add channel dimension if missing
        masks = masks.unsqueeze(1)

    masks = masks.float()
    pad = kernel_size // 2

    for _ in range(iterations):
        masks = -torch.nn.functional.max_pool2d(-masks, kernel_size=kernel_size, stride=1, padding=pad)

    return masks
