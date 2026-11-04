# Shipped example: air-hockey TTE pair

This directory holds the minimal inputs for the headline two-clip demo
(`bash scripts/run_demo.sh`). Everything is exactly what Real-VAS feeds
to the demo — no derived products, no pre-composited frames.

## The clips

| File | Shape | Frames | FPS | What it shows |
|---|---|---|---|---|
| `fg_box.mp4` | 1440×1080 | 27 | 15 | Static cardboard box on an air-hockey table, filmed from the same camera position as `bg_puck.mp4`. |
| `bg_puck.mp4` | 1440×1080 | 27 | 15 | Same table, no box — a hand slides a red puck right-to-left across the surface. |

Both clips were captured as a TTE (track-through-everything) pair: the
camera doesn't move between the two shots, so when `fg_box.mp4` is
composited over `bg_puck.mp4` the box appears to occlude the puck+hand
naturally. The puck's amodal ground-truth mask is simply its modal mask
from `bg_puck.mp4` — it's never actually hidden in the source, so the
dataset knows where it is at every frame.

## The masks

| File | What it is |
|---|---|
| `masks/fg_box_frame0.png` | First-frame (frame 0) binary mask of the box in `fg_box.mp4`. 0/255 PNG, same resolution as the clip. |
| `masks/bg_puck_frame0.png` | First-frame binary mask of the puck+hand in `bg_puck.mp4`. |

The demo passes each first-frame mask to SAM2 as the prompt and SAM2
propagates it through the rest of the clip. These masks are hand-cleaned
versions — supplying them is equivalent to `--fg-prompt "cardboard box"`
+ `--bg-prompt "hand"` via LangSAM, but faster, deterministic, and
doesn't need LangSAM installed.

## Source

Both clips came from the same recording session that produced the
`cam5_...` air-hockey clips in the Real-VAS dataset (split: `occlusion`).
They've been trimmed to 27 frames for a quick demo.
