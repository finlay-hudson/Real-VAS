"""
Real-VAS demo. Composites an occluder clip (fg) over an occludee clip (bg) and
produces modal + amodal ground-truth masks for the occludee.

Modes
-----
Two-clip (fg from clip A overlaid onto bg clip B) — minimal, no extra deps:

    python -m real_vas.demo \\
        --fg examples/fg_box.mp4  --fg-prompt "cardboard box" \\
        --bg examples/bg_puck.mp4 --bg-prompt "hand" \\
        --sam-ckpt /path/to/sam2_hiera_large.pt \\
        --longest-edge 480 --no-align \\
        --out out/smoke

Two-clip with Generative Omnimatte (clean shadows / reflections on the bg plate):

    python -m real_vas.demo \\
        --fg examples/fg_box.mp4  --fg-prompt "cardboard box" \\
        --bg examples/bg_puck.mp4 --bg-prompt "hand" \\
        --sam-ckpt /path/to/sam2_hiera_large.pt \\
        --gen-omnimatte-model-root /path/to/CogVideoX-Fun-V1.5-5b-InP \\
        --gen-omnimatte-transformer /path/to/cogvideox-v1.5-5b-transformer.safetensors \\
        --out out/smoke_omni

Single-video, user-named (both objects segmented in the same clip, fg erased
with Gen-Omnimatte before re-compositing — matches the paper's single-video flow):

    python -m real_vas.demo \\
        --video clips/scene.mp4 \\
        --fg-prompt "ball" --bg-prompt "person" \\
        --sam-ckpt /path/to/sam2_hiera_large.pt \\
        --gen-omnimatte-model-root ... --gen-omnimatte-transformer ... \\
        --out out/single_run

Optional camera-motion emulation on the composite:

    ... --camera-motion pan_right

    Presets: pan_left, pan_right, pan_on_zoomed_canvas, dutch_angle_pan,
             zoom_in_tilt_down, slow_zoom_and_pan_left, slow_zoom_and_pan_right

Outputs under --out:
    frames/00000.jpg ...         fg composited over bg
    gt_masks/00000.png ...       amodal mask of the bg object
    visible_masks/00000.png ...  modal (visible-after-occlusion) mask of bg
    debugs/mask_overlays/ ...    red = fg occluder, green = bg amodal
    debugs/see_through_bg/ ...   composite with amodal overlay
    composite.mp4                convenience video of frames/
    comparison.mp4               2x2 grid: BG | FG / Composite | Composite+amodal

Gen-Omnimatte dependency
------------------------
The `--gen-omnimatte-*` modes require the upstream `gen_omnimatte_tools`
package (containing vendored `omnimatte/` and `videox_fun/` libraries), which
is **not shipped** with this repo. Install it separately and expose it on
PYTHONPATH as `real_vas.utils.gen_omnimatte_tools`. See the README for setup.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from natsort import natsorted
from PIL import Image


# ---------------------------------------------------------------------------
# NVIDIA driver 580 (CUDA 13) workaround
# ---------------------------------------------------------------------------
# On NVIDIA driver 580.x + Ada (sm_89), torch's nvrtc JIT fails to compile
# the integer-reduction kernel used by `Tensor.prod`, surfacing as
# `RuntimeError: CUDA driver error: invalid argument` deep inside
# GroundingDINO (`spatial_shapes.prod(1)`). Float `prod` and all other ops
# work fine; only integer `prod` on CUDA breaks. We intercept the method on
# the Tensor class and bounce int cases through CPU. The shape tensor is
# tiny, so the cost is microseconds.
#
# Safe to delete this block once PyTorch / the driver stop reproducing the
# bug — the production code paths all use CUDA directly.
# ---------------------------------------------------------------------------
_orig_prod = torch.Tensor.prod


def _safe_prod(self, *args, **kwargs):
    if self.is_cuda and self.dtype in (torch.int64, torch.int32, torch.int16, torch.int8):
        return _orig_prod(self.cpu(), *args, **kwargs).to(self.device)
    return _orig_prod(self, *args, **kwargs)


torch.Tensor.prod = _safe_prod

from real_vas.utils.segmentation_utils import (
    LangSamRunner,
    SAMVid,
    get_sam2_predictor,
)
from real_vas.utils.video_loader import VideoBackend, VideoLoader
from real_vas.utils.vis_utils import CompositeMaker, overlay_mask_over_image


def _load_clip(path: Path, longest_edge: int) -> tuple[np.ndarray, float]:
    """Load a whole clip as (T, H, W, 3) uint8 RGB and return its fps."""
    with VideoLoader(str(path), backend=VideoBackend.DECORD, longest_edge=longest_edge) as vid:
        total = vid.get_num_frames()
        fps = vid.get_fps()
        frames, _ = vid.get_frame_generator(
            frame_indices=list(range(total)), chunk_size=-1, just_return=True,
        )
    return np.asarray(frames), float(fps)


def _match_bg_to_fg(bg_frames: np.ndarray, fg_frames: np.ndarray) -> np.ndarray:
    """Trim or loop bg to fg length, then resize spatially to fg's (H, W)."""
    t_fg, h, w, _ = fg_frames.shape
    t_bg = bg_frames.shape[0]
    if t_bg >= t_fg:
        bg_frames = bg_frames[:t_fg]
    else:
        reps = (t_fg + t_bg - 1) // t_bg
        bg_frames = np.tile(bg_frames, (reps, 1, 1, 1))[:t_fg]
    if bg_frames.shape[1:3] != (h, w):
        bg_frames = np.stack([cv2.resize(f, (w, h)) for f in bg_frames])
    return bg_frames


def _align_fg_to_bg_centroids(fg_frames: np.ndarray, fg_masks: np.ndarray,
                              bg_masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Shift fg frame + mask so the fg object's centroid tracks the bg object's."""
    aligned_frames = np.zeros_like(fg_frames)
    aligned_masks = np.zeros_like(fg_masks)
    for i in range(len(fg_frames)):
        fg_m = fg_masks[i] > 0
        bg_m = bg_masks[i] > 0
        if not fg_m.any() or not bg_m.any():
            aligned_frames[i] = fg_frames[i]
            aligned_masks[i] = fg_masks[i]
            continue
        fy, fx = np.argwhere(fg_m).mean(axis=0)
        by, bx = np.argwhere(bg_m).mean(axis=0)
        dy, dx = int(round(by - fy)), int(round(bx - fx))

        sf = np.roll(fg_frames[i], (dy, dx), axis=(0, 1))
        sm = np.roll(fg_masks[i], (dy, dx), axis=(0, 1))
        if dy > 0:
            sf[:dy] = 0
            sm[:dy] = 0
        elif dy < 0:
            sf[dy:] = 0
            sm[dy:] = 0
        if dx > 0:
            sf[:, :dx] = 0
            sm[:, :dx] = 0
        elif dx < 0:
            sf[:, dx:] = 0
            sm[:, dx:] = 0
        aligned_frames[i] = sf
        aligned_masks[i] = sm
    return aligned_frames, aligned_masks


def _first_frame_mask(sam_image_pred, frame_rgb: np.ndarray, prompt: str,
                      device: torch.device) -> np.ndarray:
    runner = LangSamRunner(image_sam=sam_image_pred)
    runner.to_device(device)
    img = torch.from_numpy(frame_rgb[None]).to(device)
    results = runner.run_custom(img, prompt=[prompt])
    if not results:
        raise RuntimeError(f"LangSAM returned no detections for prompt {prompt!r}")
    r = results[0]
    best = int(torch.argmax(r["scores"]))
    mask = r["masks"][best]
    if mask.ndim == 3:
        mask = mask[0]
    return (mask.cpu().numpy() > 0).astype(np.uint8) * 255


def _propagate_mask(sam_vid: SAMVid, frames: np.ndarray, first_mask: np.ndarray,
                    device: torch.device) -> np.ndarray:
    h, w = frames.shape[1:3]
    vid_t = torch.from_numpy(frames)
    mask_t = torch.from_numpy(first_mask > 0).bool()
    masks = sam_vid(vid_t, mask_t, device=device, resize_to=(h, w))
    masks = masks.permute(1, 0, 2, 3)
    return (masks[0].cpu().numpy().astype(np.uint8)) * 255


def _segment_clip(sam_image_pred, sam_vid: SAMVid, frames: np.ndarray,
                  prompt: str, device: torch.device) -> np.ndarray:
    first = _first_frame_mask(sam_image_pred, frames[0], prompt, device)
    return _propagate_mask(sam_vid, frames, first, device)


def _load_first_frame_mask(path: Path, frames: np.ndarray) -> np.ndarray:
    """Read a user-supplied first-frame mask PNG, resize to the loaded frame
    size, and return as (H, W) uint8 0/255."""
    mask = np.array(Image.open(path).convert("L"))
    h, w = frames.shape[1:3]
    if mask.shape != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return ((mask > 127).astype(np.uint8)) * 255


def _segment_clip_from_mask(sam_vid: SAMVid, frames: np.ndarray,
                            first_mask: np.ndarray, device: torch.device) -> np.ndarray:
    return _propagate_mask(sam_vid, frames, first_mask, device)


def _write_mp4(path: Path, frames_rgb: np.ndarray, fps: float) -> None:
    h, w = frames_rgb.shape[1:3]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames_rgb:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()


def _label_panel(frame_rgb: np.ndarray, label: str) -> np.ndarray:
    """Overlay a short label in the top-left of a panel with a dark backing."""
    out = frame_rgb.copy()
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    cv2.rectangle(out, (0, 0), (tw + 20, th + 20), (0, 0, 0), -1)
    cv2.putText(out, label, (10, th + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (255, 255, 255), 2, cv2.LINE_AA)
    return out


def _write_comparison_mp4(path: Path, bg_frames: np.ndarray, fg_frames: np.ndarray,
                          composite_frames: np.ndarray, bg_masks: np.ndarray,
                          fps: float, panel_width: int = 640) -> None:
    """2x2 grid: BG | FG / Composite | Composite+Amodal. Written at `path`."""
    t, _, _, _ = composite_frames.shape[:4] if composite_frames.ndim == 4 else (
        composite_frames.shape[0], *composite_frames.shape[1:])

    def _resize(frames: np.ndarray) -> np.ndarray:
        h, w = frames.shape[1:3]
        new_h = int(round(panel_width * h / w))
        return np.stack([cv2.resize(f, (panel_width, new_h)) for f in frames])

    bg_r = _resize(bg_frames)
    fg_r = _resize(fg_frames)
    comp_r = _resize(composite_frames)
    overlay = np.stack([
        overlay_mask_over_image(c, m, alpha=0.45, color=(0, 255, 0))
        for c, m in zip(composite_frames, bg_masks)
    ])
    overlay_r = _resize(overlay)

    panels = [bg_r, fg_r, comp_r, overlay_r]
    labels = ["BG clip", "FG clip", "Composite", "Composite + amodal"]
    panels = [np.stack([_label_panel(f, lbl) for f in p]) for p, lbl in zip(panels, labels)]

    top = np.concatenate([panels[0], panels[1]], axis=2)
    bot = np.concatenate([panels[2], panels[3]], axis=2)
    grid = np.concatenate([top, bot], axis=1)
    _write_mp4(path, grid, fps)


def _run_gen_omnimatte_cleanup(frames: np.ndarray, fg_masks: np.ndarray,
                               save_path: Path, model_name: Path,
                               transformer_path: Path) -> np.ndarray:
    """Erase the fg object from `frames` with GenOmnimatte; return (T, H, W, 3) uint8.

    Lazy-imports `real_vas.utils.gen_omnimatte_tools` — users must install it
    separately (see the README). Raises a clear error if unavailable.
    """
    try:
        from real_vas.utils.gen_omnimatte_tools.configs import get_config
        from real_vas.utils.gen_omnimatte_tools.runner import (
            get_components,
            run_through_gen_omnimatte,
        )
    except ImportError as e:
        raise SystemExit(
            "Gen-Omnimatte support requires `real_vas.utils.gen_omnimatte_tools`, "
            "which is not shipped with this repo. See the README section "
            "'Gen-Omnimatte setup' for install instructions.\n"
            f"Original error: {e}"
        ) from e

    config = get_config()
    pipeline, vae, generator = get_components(config, str(model_name), str(transformer_path))

    front_masks = fg_masks[None]
    run_through_gen_omnimatte(
        config, pipeline, vae, generator, frames, front_masks,
        prompt="a video of a scene", save_path=save_path,
        keep_fg_ids=[-1], save_first_stage=True,
    )

    first_stage_dir = save_path / "first_stage_-1"
    png_files = natsorted(first_stage_dir.glob("*.png"))
    if not png_files:
        raise RuntimeError(f"GenOmnimatte produced no frames in {first_stage_dir}")
    clean = np.stack([np.array(Image.open(p).convert("RGB")) for p in png_files])

    orig_h, orig_w = frames.shape[1:3]
    if clean.shape[1:3] != (orig_h, orig_w):
        clean = np.stack([cv2.resize(f, (orig_w, orig_h)) for f in clean])
    return clean[: len(frames)]


def _run_two_clip(args, device, sam_image_pred, sam_vid):
    print(f"[1/4] Loading clips (longest_edge={args.longest_edge})")
    fg_frames, fg_fps = _load_clip(args.fg, args.longest_edge)
    bg_frames, _ = _load_clip(args.bg, args.longest_edge)
    bg_frames = _match_bg_to_fg(bg_frames, fg_frames)
    print(f"      fg: {fg_frames.shape} @ {fg_fps:.2f} fps, bg: {bg_frames.shape}")

    if args.fg_mask_png is not None:
        print(f"[2/4] Using supplied fg mask {args.fg_mask_png}")
        first_fg = _load_first_frame_mask(args.fg_mask_png, fg_frames)
        fg_masks = _segment_clip_from_mask(sam_vid, fg_frames, first_fg, device)
    else:
        print(f"[2/4] Segmenting fg prompt {args.fg_prompt!r}")
        fg_masks = _segment_clip(sam_image_pred, sam_vid, fg_frames, args.fg_prompt, device)

    if args.bg_mask_png is not None:
        print(f"[3/4] Using supplied bg mask {args.bg_mask_png}")
        first_bg = _load_first_frame_mask(args.bg_mask_png, bg_frames)
        bg_masks = _segment_clip_from_mask(sam_vid, bg_frames, first_bg, device)
    else:
        print(f"[3/4] Segmenting bg prompt {args.bg_prompt!r}")
        bg_masks = _segment_clip(sam_image_pred, sam_vid, bg_frames, args.bg_prompt, device)

    if not args.no_align:
        print("[align] shifting fg so its mask centroid tracks the bg mask centroid")
        fg_frames, fg_masks = _align_fg_to_bg_centroids(fg_frames, fg_masks, bg_masks)

    if args.gen_omnimatte_model_root is not None and args.gen_omnimatte_transformer is not None:
        print("[clean] GenOmnimatte: erasing fg from bg plate before compositing")
        gen_dir = args.out / "gen_omnimatte"
        gen_dir.mkdir(parents=True, exist_ok=True)
        bg_frames = _run_gen_omnimatte_cleanup(
            bg_frames, bg_masks, gen_dir,
            args.gen_omnimatte_model_root, args.gen_omnimatte_transformer,
        )

    out_fps = args.fps if args.fps is not None else fg_fps
    return fg_frames, fg_masks, bg_frames, bg_masks, out_fps


def _run_single_video(args, device, sam_image_pred, sam_vid):
    if args.gen_omnimatte_model_root is None or args.gen_omnimatte_transformer is None:
        raise SystemExit(
            "--gen-omnimatte-model-root and --gen-omnimatte-transformer are required "
            "in single-video mode (the fg must be erased before re-compositing)."
        )

    print(f"[load] clip (longest_edge={args.longest_edge})")
    full_frames, fps = _load_clip(args.video, args.longest_edge)
    print(f"       clip: {full_frames.shape} @ {fps:.2f} fps")

    print(f"[seg] user-named fg={args.fg_prompt!r} bg={args.bg_prompt!r}")
    fg_masks = _segment_clip(sam_image_pred, sam_vid, full_frames, args.fg_prompt, device)
    bg_masks = _segment_clip(sam_image_pred, sam_vid, full_frames, args.bg_prompt, device)

    print("[clean] GenOmnimatte: erasing fg from clip to make a clean bg plate")
    gen_dir = args.out / "gen_omnimatte"
    gen_dir.mkdir(parents=True, exist_ok=True)
    clean_bg = _run_gen_omnimatte_cleanup(
        full_frames, fg_masks, gen_dir,
        args.gen_omnimatte_model_root, args.gen_omnimatte_transformer,
    )

    out_fps = args.fps if args.fps is not None else fps
    return full_frames, fg_masks, clean_bg, bg_masks, out_fps


def _apply_camera_motion(fg_frames, fg_masks, bg_frames, bg_masks, preset_name):
    """Apply a named camera-motion preset to all four aligned stacks.

    The composited scene is warped as a single camera motion — fg frames,
    fg masks, and bg masks all transform under the same per-frame homography.
    """
    from real_vas.utils.camera_motion_tools.components import PRESETS
    from real_vas.utils.camera_motion_tools.utils import create_camera_motion

    if preset_name not in PRESETS:
        raise SystemExit(f"Unknown camera-motion preset {preset_name!r}. "
                         f"Available: {sorted(PRESETS)}")
    print(f"[camera] applying motion preset: {preset_name}")

    fg_frames_w, fg_masks_w, bg_masks_w, _, _ = create_camera_motion(
        fg_frames, fg_masks, bg_masks, selected_presets=preset_name,
    )
    bg_frames_w, _, _, _, _ = create_camera_motion(
        bg_frames, fg_masks, bg_masks, selected_presets=preset_name,
    )
    return fg_frames_w, fg_masks_w, bg_frames_w, bg_masks_w


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--video", type=Path,
                      help="Single-video mode: fg and bg objects both come from this clip. "
                           "Requires --fg-prompt, --bg-prompt, and the --gen-omnimatte-* paths.")
    mode.add_argument("--fg", type=Path,
                      help="Two-clip mode: foreground clip (the occluder).")
    p.add_argument("--bg", type=Path,
                   help="Two-clip mode: background clip (the occludee). Required with --fg.")
    p.add_argument("--fg-prompt", default=None,
                   help="LangSAM prompt for the fg object (occluder). "
                        "Required unless --fg-mask-png is given.")
    p.add_argument("--bg-prompt", default=None,
                   help="LangSAM prompt for the bg object (occludee). "
                        "Required unless --bg-mask-png is given.")
    p.add_argument("--fg-mask-png", type=Path, default=None,
                   help="Pre-computed first-frame mask (PNG, 0/255) for the fg object. "
                        "If given, skips LangSAM for the fg side.")
    p.add_argument("--bg-mask-png", type=Path, default=None,
                   help="Pre-computed first-frame mask (PNG, 0/255) for the bg object. "
                        "If given, skips LangSAM for the bg side.")
    p.add_argument("--out", type=Path, required=True, help="Output directory.")
    p.add_argument("--sam-ckpt", type=Path, required=True,
                   help="Path to SAM2 checkpoint, e.g. sam2_hiera_large.pt.")
    p.add_argument("--longest-edge", type=int, default=720,
                   help="Resize clips so the longest edge is this many px.")
    p.add_argument("--fps", type=float, default=None,
                   help="Output fps; defaults to the source clip's fps.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-align", action="store_true",
                   help="Two-clip mode: skip centroid-aligning the fg object onto the bg object.")
    p.add_argument("--gen-omnimatte-model-root", type=Path, default=None,
                   help="HF snapshot dir for CogVideoX-Fun-V1.5-5b-InP. "
                        "Required for --video mode; optional for --fg/--bg to clean the bg plate.")
    p.add_argument("--gen-omnimatte-transformer", type=Path, default=None,
                   help="cogvideox-v1.5-5b-transformer.safetensors path.")
    p.add_argument("--camera-motion", default=None,
                   help="Apply a camera-motion preset to the composite. See module docstring for list.")
    args = p.parse_args()

    if args.fg is not None and args.bg is None:
        p.error("--fg requires --bg")
    if args.fg is not None:
        if args.fg_prompt is None and args.fg_mask_png is None:
            p.error("two-clip mode: supply either --fg-prompt or --fg-mask-png")
        if args.bg_prompt is None and args.bg_mask_png is None:
            p.error("two-clip mode: supply either --bg-prompt or --bg-mask-png")

    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    sam_image_pred = get_sam2_predictor(
        single_image=True, checkpoint=str(args.sam_ckpt), device="cpu",
    )
    sam_vid = SAMVid(str(args.sam_ckpt))

    if args.video is not None:
        print(f"[mode] single-video: {args.video}")
        fg_frames, fg_masks, composite_bg, bg_masks, out_fps = _run_single_video(
            args, device, sam_image_pred, sam_vid,
        )
    else:
        print(f"[mode] two-clip: fg={args.fg} bg={args.bg}")
        fg_frames, fg_masks, composite_bg, bg_masks, out_fps = _run_two_clip(
            args, device, sam_image_pred, sam_vid,
        )

    if args.camera_motion:
        fg_frames, fg_masks, composite_bg, bg_masks = _apply_camera_motion(
            fg_frames, fg_masks, composite_bg, bg_masks, args.camera_motion,
        )

    print(f"[compose] fg={fg_frames.shape} bg={composite_bg.shape} fg_mask={fg_masks.shape}")
    compositer = CompositeMaker()
    comb_ims = compositer([fg_frames], composite_bg, [fg_masks])
    compositer.image_saver(comb_ims, fg_masks, bg_masks, out_dir=args.out, with_debugs=True)

    out_rgb = np.stack([c[0] if c.ndim == 4 else c for c in comb_ims])
    _write_mp4(args.out / "composite.mp4", out_rgb, out_fps)
    _write_comparison_mp4(
        args.out / "comparison.mp4",
        bg_frames=composite_bg, fg_frames=fg_frames,
        composite_frames=out_rgb, bg_masks=bg_masks,
        fps=out_fps,
    )
    print(f"Done. {len(out_rgb)} frames written to {args.out}")


if __name__ == "__main__":
    main()
