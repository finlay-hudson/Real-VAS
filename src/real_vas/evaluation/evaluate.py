"""
Real-VAS evaluation CLI.

Scores amodal-segmentation predictions against the Real-VAS dataset using the
same metrics as the paper: mIoU (all frames), mIoUfo (any-occlusion frames),
mIoUffo (fully-occluded frames), mIoUocc (hidden-region IoU on
partially-occluded frames).

Expected prediction layout (one file per source-frame, mirroring the dataset
tree so every video has its own predictions dir):

    predictions/
    ├── containment/
    │   └── <vid_name>/
    │       └── pred_masks/     # single-generation
    │           ├── 00000.png
    │           ├── 00001.png
    │           └── ...
    │       # or, for multi-generation:
    │       └── gen_0/00000.png ...
    │       └── gen_1/00000.png ...
    └── occlusion/
        └── ...

Masks can be grayscale PNG with any value > 0 taken as foreground, or
strictly 0/255. Shape must match the dataset's modal/amodal masks.

Example:
    python -m real_vas.evaluation.evaluate \\
        --dataset ~/data/Real-VAS \\
        --predictions ~/my_model_outputs/ \\
        --split occlusion \\
        --output results.json
"""

import argparse
import json
from pathlib import Path

import torch
from natsort import natsorted
from PIL import Image
from torchvision import transforms

from real_vas.evaluation.dataset import DataConfigREALVAS, DatasetSplit, REALVASDataset
from real_vas.evaluation.eval_runner import get_iou_results
from real_vas.utils.occlusion_utils import OcclusionInfo


_PRED_SUBDIRS_SINGLE = ["pred_masks", "amodal_masks", "predictions"]


def _load_pred_masks(pred_dir: Path, h: int, w: int) -> torch.Tensor:
    """Load a prediction directory into a (num_generations, T, H, W) uint8 tensor.

    Accepts two layouts under `pred_dir`:
      * flat PNGs → single generation
      * `gen_0/`, `gen_1/`, ... each holding PNGs → multi-generation
    """
    tx = transforms.ToTensor()

    def _stack(d: Path) -> torch.Tensor:
        pngs = natsorted(d.glob("*.png"))
        if not pngs:
            raise FileNotFoundError(f"No PNGs in {d}")
        masks = torch.stack([tx(Image.open(p).convert("L")) for p in pngs])  # (T, 1, H, W) float [0,1]
        masks = (masks.squeeze(1) > 0.5).to(torch.uint8) * 255
        if masks.shape[1:] != (h, w):
            raise ValueError(
                f"Prediction shape {tuple(masks.shape)} does not match GT (T, {h}, {w}) at {d}"
            )
        return masks

    gen_dirs = natsorted(
        d for d in pred_dir.iterdir()
        if d.is_dir() and d.name.startswith("gen_")
    )
    if gen_dirs:
        return torch.stack([_stack(d) for d in gen_dirs])
    return _stack(pred_dir).unsqueeze(0)


def _find_pred_dir(pred_root: Path, vid_rel: Path) -> Path:
    """Locate `<pred_root>/<vid_rel>/{pred_masks|amodal_masks|...}/` or the vid dir itself."""
    base = pred_root / vid_rel
    if not base.exists():
        raise FileNotFoundError(f"Missing predictions for {vid_rel} under {pred_root}")
    for sub in _PRED_SUBDIRS_SINGLE:
        if (base / sub).is_dir():
            return base / sub
    # fall back to the vid dir itself (holds PNGs or gen_* subdirs directly)
    return base


def _aggregate(per_video: dict) -> dict:
    """NaN-safe mean of each metric across videos (per-generation).

    Matches `np.nanmean`: divides by the number of non-NaN values, NOT the
    total video count. Critical for `mIOUffo` and `mIOUocc`, which are NaN
    on videos that have no fully-occluded / partially-occluded frames —
    counting those as zero would artificially drag the aggregate down.
    """
    out = {}
    all_gens = sorted({k for v in per_video.values() for k in v})
    for gen_key in all_gens:
        rows = [v[gen_key] for v in per_video.values() if gen_key in v]
        metrics = rows[0].keys()
        out[gen_key] = {}
        for m in metrics:
            valid = [r[m] for r in rows if r[m] == r[m]]
            out[gen_key][m] = float(sum(valid) / len(valid)) if valid else float("nan")
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset", type=Path, required=True,
                   help="Root of the Real-VAS dataset (containing containment/ and occlusion/).")
    p.add_argument("--predictions", type=Path, required=True,
                   help="Prediction root, mirroring the dataset split layout.")
    p.add_argument("--split", choices=["containment", "occlusion", "both"], default="both")
    p.add_argument("--output", type=Path, default=None,
                   help="Optional JSON file to write per-video + aggregate metrics.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)

    split_map = {
        "containment": DatasetSplit.CONTAINMENT,
        "occlusion": DatasetSplit.OCCLUSION,
        "both": DatasetSplit.BOTH,
    }
    cfg = DataConfigREALVAS(data_root=args.dataset, frame_dir_name=None)
    ds = REALVASDataset(cfg, dataset_split=split_map[args.split])
    print(f"Loaded Real-VAS ({args.split}) — {len(ds)} videos")

    per_video: dict[str, dict] = {}
    missing: list[str] = []
    skipped_bad_anno: list[str] = []
    for i in range(len(ds)):
        _, _, amodal_masks, modal_masks, anno, vid_name = ds[i]
        if amodal_masks is None:
            raise RuntimeError(f"Dataset config disabled amodal masks; cannot evaluate ({vid_name})")

        try:
            pred_dir = _find_pred_dir(args.predictions, Path(vid_name))
        except FileNotFoundError:
            missing.append(vid_name)
            continue

        # Dataset tensors come from ToTensor() → float32 in [0, 1]; collapse to (T, H, W) uint8 0/255.
        gt_amodal = (amodal_masks.squeeze(1) > 0.5).to(torch.uint8).to(device) * 255
        gt_vis = (modal_masks.squeeze(1) > 0.5).to(torch.uint8).to(device) * 255

        pred = _load_pred_masks(pred_dir, gt_amodal.shape[1], gt_amodal.shape[2]).to(device)

        occl_info = [OcclusionInfo.from_dict(o) for o in anno.get("occlusion", [])]
        if len(occl_info) != gt_amodal.shape[0]:
            print(f"  [skip] {vid_name}: {len(occl_info)} occlusion entries vs "
                  f"{gt_amodal.shape[0]} GT frames — annotation length mismatch")
            skipped_bad_anno.append(vid_name)
            continue

        per_video[vid_name] = get_iou_results(pred, gt_amodal, gt_vis, occl_info)
        print(f"  [{i + 1}/{len(ds)}] {vid_name}: {per_video[vid_name]['gen: 0']['mIOU']:.3f} mIoU")

    if missing:
        print(f"\n{len(missing)} videos skipped (no predictions):")
        for v in missing[:5]:
            print(f"  - {v}")
        if len(missing) > 5:
            print(f"  (+ {len(missing) - 5} more)")
    if skipped_bad_anno:
        print(f"\n{len(skipped_bad_anno)} videos skipped (annotation length ≠ frame count):")
        for v in skipped_bad_anno[:5]:
            print(f"  - {v}")
        if len(skipped_bad_anno) > 5:
            print(f"  (+ {len(skipped_bad_anno) - 5} more)")

    agg = _aggregate(per_video)
    print("\n=== Aggregate ===")
    for gen_key, metrics in agg.items():
        parts = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        print(f"  {gen_key}: {parts}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "split": args.split,
            "num_evaluated": len(per_video),
            "num_missing": len(missing),
            "num_skipped_bad_anno": len(skipped_bad_anno),
            "skipped_bad_anno": skipped_bad_anno,
            "aggregate": agg,
            "per_video": per_video,
        }, indent=2))
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
