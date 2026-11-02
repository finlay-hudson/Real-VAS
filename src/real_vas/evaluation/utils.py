import torch


def batch_iou(masks1: torch.Tensor, masks2: torch.Tensor) -> torch.Tensor:
    """
    Calculates the IoU between two batches of binary masks using PyTorch.

    Args:
        masks1: A batch of masks. Shape: (..., H, W).
        masks2: Another batch of masks. Shape: (..., H, W).

    Returns:
        IoU scores for each pair of masks. Shape: (...).
    """
    assert masks1.shape == masks2.shape, "Mask batches must have the same shape"

    # Identify masks that contain NaN values, which should be ignored
    nan_mask = (torch.isnan(masks1).flatten(start_dim=-2).any(dim=-1) |
                torch.isnan(masks2).flatten(start_dim=-2).any(dim=-1))

    # Ensure masks are boolean for logical operations
    masks1_bool = masks1.bool()
    masks2_bool = masks2.bool()

    # Calculate intersection and union over the spatial dimensions (H, W)
    intersection = torch.logical_and(masks1_bool, masks2_bool).sum(dim=(-1, -2))
    union = torch.logical_or(masks1_bool, masks2_bool).sum(dim=(-1, -2))

    # Calculate IoU with protection against division by zero
    iou_scores = torch.where(
        union > 0,
        intersection.float() / union.float(),
        torch.zeros_like(intersection, dtype=torch.float)
    )

    # Re-apply NaN mask for ignored elements
    iou_scores[nan_mask] = float('nan')

    return iou_scores
