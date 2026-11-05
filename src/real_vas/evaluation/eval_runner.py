from typing import List

import torch

from real_vas.evaluation.utils import batch_iou
from real_vas.utils.occlusion_utils import OcclusionInfo, OcclusionLevel


def get_iou_results(exp_pred_masks: torch.Tensor, gt_amodal_masks: torch.Tensor, gt_vis_masks: torch.Tensor,
                    gt_occlusion_info: List[OcclusionInfo]) -> dict:
    """
    Calculates various IoU metrics between predicted and ground truth masks using PyTorch.

    Returns per-generation dict of {mIOU, mIOUffo, mIOUfo, mIOUocc}. Empty
    buckets (e.g. a video with no fully-occluded frames) return NaN for that
    metric — the CLI's aggregator (`_aggregate` in `evaluate.py`) is NaN-safe
    and matches `np.nanmean` across videos.
    """
    device = exp_pred_masks.device
    num_generations = exp_pred_masks.shape[0]

    # Normalize masks from [0, 255] to [0, 1] if necessary.
    if exp_pred_masks.max() > 1:
        exp_pred_masks = exp_pred_masks // 255
    if gt_amodal_masks.max() > 1:
        gt_amodal_masks = gt_amodal_masks.clone() // 255
    if gt_vis_masks.max() > 1:
        gt_vis_masks = gt_vis_masks.clone() // 255

    # Expand ground truth masks for batch processing
    gt_amodal_expanded = gt_amodal_masks.unsqueeze(0).expand_as(exp_pred_masks)
    gt_vis_expanded = gt_vis_masks.unsqueeze(0).expand_as(exp_pred_masks)

    # 1. Calculate standard amodal IoU
    all_ious = batch_iou(gt_amodal_expanded, exp_pred_masks)

    # 2. Calculate IoU for the non-visible (occluded) parts
    non_vis_mask_gt = (gt_amodal_expanded.float() - gt_vis_expanded.float()).clamp(min=0)
    non_vis_mask_pred = (exp_pred_masks.float() - gt_vis_expanded.float()).clamp(min=0)
    all_non_vis_iou = batch_iou(non_vis_mask_gt, non_vis_mask_pred)

    # 3. Create boolean tensors to select frames with specific properties
    fully_occluded_frames = gt_vis_masks.view(gt_vis_masks.shape[0], -1).sum(dim=-1) < 1
    any_occluded_frames = torch.tensor(
        [o.level != OcclusionLevel.NO_OCCLUSION for o in gt_occlusion_info],
        dtype=torch.bool, device=device
    )
    frames_with_non_vis_parts = (gt_amodal_masks - gt_vis_masks).view(gt_vis_masks.shape[0], -1).sum(dim=-1) > 0

    # 4. Aggregate results for each generation
    iou_res = {}
    for gen in range(num_generations):
        ious = all_ious[gen]
        non_vis_ious = all_non_vis_iou[gen]

        iou_res[f"gen: {gen}"] = {
            "mIOU": ious.mean().item(),
            "mIOUffo": ious[fully_occluded_frames].mean().item(),
            "mIOUfo": ious[any_occluded_frames].mean().item(),
            "mIOUocc": non_vis_ious[frames_with_non_vis_parts].mean().item()
        }

    return iou_res
