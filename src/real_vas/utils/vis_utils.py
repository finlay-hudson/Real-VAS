import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch


def overlay_mask_over_image(image, mask, alpha=0.5, color=(0, 255, 0)):
    # Ensure mask and image have the same dimensions
    if image.shape[:2] != mask.shape[:2]:
        raise ValueError("Image and mask dimensions do not match.")
    if not mask.any():
        return image
    if mask.max() > 1:
        mask = mask // 255
    overlay_mask = np.zeros_like(image)
    overlay_mask[mask == 1] = color

    anno_img = image.copy()
    anno_img[mask == 1] = cv2.addWeighted(image[mask == 1], 1 - alpha, overlay_mask[mask == 1], alpha, 0)

    return anno_img


def add_text(frame, txt, pos=(0, 0), font=cv2.FONT_HERSHEY_PLAIN, scale=1.0, thickness=1, txt_col=(0, 0, 0),
             bg_col=None):
    was_float = False
    if frame.dtype.kind == 'f':
        was_float = True
        frame = (frame * 255).astype(np.uint8)
    x, y = pos
    text_size, _ = cv2.getTextSize(txt, font, scale, thickness)
    text_w, text_h = text_size
    if bg_col is not None:
        cv2.rectangle(frame, (pos[0] - 5, pos[1] - 5), (x + text_w + 5, y + text_h + 5), bg_col, -1)
    cv2.putText(frame, txt, (x, int(y + text_h + 1)), font, scale, txt_col, thickness)

    if was_float:
        return frame / 255.0

    return frame


class CompositeMaker(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def blend_multiple_foregrounds(im_bg, im_fgs, alpha_fgs, ref_im):
        """
        Blends multiple foreground images onto a background using their alpha masks.

        Parameters:
        - im_bg: Background image (H, W, C)
        - im_fgs: List of foreground images [(H, W, C), ...]
        - alpha_fgs: List of alpha masks [(H, W), ...]
        - ref_im: Reference image (H, W, C)

        Returns:
        - Blended image (H, W, C)
        """
        composite = im_bg.copy()
        alpha_accumulated = np.zeros_like(alpha_fgs[0])  # Initialize accumulated alpha

        for im_fg, alpha_fg in zip(im_fgs, alpha_fgs):
            # Compute corrected foreground color
            F_i = np.clip((im_fg - (1 - alpha_fg) * ref_im) / np.maximum(alpha_fg, 1e-6), 0, 1)

            # Blend using accumulated alpha (corrected)
            composite = composite * (1 - alpha_fg) + F_i * alpha_fg * (1 - alpha_accumulated)

            # Update accumulated alpha properly
            alpha_accumulated = alpha_accumulated + alpha_fg * (1 - alpha_accumulated)

        return composite[..., ::-1]

    @staticmethod
    def process_ims(ims):
        processed_ims = []
        for im in ims:
            if isinstance(im, Image.Image):
                im = np.array(im)
            elif isinstance(im, torch.Tensor):
                im = im.cpu().detach().numpy()
            p_im = im.astype(np.float32)
            if p_im.max() > 1.0:
                p_im = p_im / 255
            if p_im.ndim == 2:
                p_im = p_im[:, :, np.newaxis]
            processed_ims.append(p_im)

        return processed_ims

    def image_saver(self, comb_ims, fg_alphas, bg_alphas, out_dir, with_debugs=True):
        if isinstance(fg_alphas, torch.Tensor):
            fg_alphas = fg_alphas.cpu().numpy()
        if isinstance(bg_alphas, torch.Tensor):
            bg_alphas = bg_alphas.cpu().numpy()
        if fg_alphas.ndim == 4:
            fg_alphas = fg_alphas[..., 0]
        if bg_alphas.ndim == 4:
            bg_alphas = bg_alphas[..., 0]
        out_dir_composite = out_dir / "frames"
        out_dir_composite.mkdir(exist_ok=True)
        out_dir_gt = out_dir / "gt_masks"
        out_dir_gt.mkdir(exist_ok=True, parents=True)
        out_dir_vis_masks = out_dir / "visible_masks"
        out_dir_vis_masks.mkdir(exist_ok=True, parents=True)
        out_dir_debug_mask_overlays = out_dir / "debugs" / "mask_overlays"
        out_dir_debug_see_through = out_dir / "debugs" / "see_through_bg"
        if with_debugs:
            out_dir_debug_mask_overlays.mkdir(exist_ok=True, parents=True)
            out_dir_debug_see_through.mkdir(exist_ok=True, parents=True)
        for i, (fg_alpha, bg_alpha, comb_im) in enumerate(zip(fg_alphas, bg_alphas, comb_ims)):
            Image.fromarray(comb_im[0]).save(out_dir_composite / f"{str(i).zfill(5)}.jpg")
            vis_mask = (bg_alpha & (fg_alpha == 0) * 255).astype(np.uint8)
            Image.fromarray(bg_alpha).save(out_dir_gt / f"{str(i).zfill(5)}.png")
            Image.fromarray(vis_mask).save(out_dir_vis_masks / f"{str(i).zfill(5)}.png")
            if with_debugs:
                vis_mask = np.zeros((*fg_alpha.shape[:2], 3), dtype=np.uint8)
                vis_mask[fg_alpha > 0] = [255, 0, 0]  # Red for front object
                vis_mask[bg_alpha > 0] = [0, 255, 0]  # Green for back object
                Image.fromarray(vis_mask).save(out_dir_debug_mask_overlays / f"{i}.png")
                overlay_mask = overlay_mask_over_image(comb_im[0], bg_alpha)
                # TODO if we have occlusion info
                # add_text(overlay_mask, f"Occl: {occl_infos[i].level.name} - {occl_infos[i].amount:.3f}", (10, 10),
                #          scale=3, thickness=2, bg_col=(255, 255, 255))
                Image.fromarray(overlay_mask).save(out_dir_debug_see_through / f"{i}.jpg")

    def forward(self, fg_ims: list[list], bg_ims: list, fg_alphas: list[list]):
        """
        Args:
            fg_ims: list of list like object
            bg_ims: list like object
            fg_alphas: list of list like object

        Returns:

        """
        # TODO THIS is taken from composite code and should not be repeated
        # p_fg_ims = [self.process_ims(fg_im) for fg_im in fg_ims]
        # p_fg_alphas = [self.process_ims(fg_alpha) for fg_alpha in fg_alphas]
        comb_ims = []
        for m_idx, bg_im in enumerate(bg_ims):
            p_bg_im = self.process_ims([bg_im])
            if m_idx >= len(fg_ims[0]):
                break
            current_fgs = self.process_ims([fg_list[m_idx] for fg_list in fg_ims])
            current_alphas = self.process_ims([alpha_list[m_idx] for alpha_list in fg_alphas])
            comb_im = self.blend_multiple_foregrounds(p_bg_im, current_fgs, current_alphas, p_bg_im)
            comb_im = (comb_im * 255).astype(np.uint8)
            comb_im = comb_im[..., ::-1]
            comb_ims.append(comb_im)

            # anno_occlusions.append(asdict(occl_infos[i]))

        return comb_ims


class Cols:
    def __init__(self):
        cmap = plt.get_cmap("tab20")
        self.colours = [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b, _ in (cmap(i) for i in range(cmap.N))]

    def get_col(self, idx: int):
        return self.colours[idx % len(self.colours)]


def optimal_grid(num_images):
    cols = round(np.sqrt(num_images))
    rows = np.ceil(num_images / cols)

    return int(rows), int(cols)


def create_mosaic(images, rows=None, cols=None):
    if rows is None or cols is None:
        rows, cols = optimal_grid(len(images))
    # Assuming all images are of the same size
    img_height, img_width, img_channels = images[0].shape

    # Create a blank canvas
    mosaic = np.zeros((rows * img_height, cols * img_width, img_channels), dtype=images[0].dtype)

    # Place each image in the mosaic
    for i, img in enumerate(images):
        row = i // cols
        col = i % cols
        mosaic[row * img_height:(row + 1) * img_height, col * img_width:(col + 1) * img_width, :] = img

    return mosaic


def add_border(f, border_ceof=3, col=(255, 255, 255)):
    return cv2.copyMakeBorder(f.copy(), border_ceof, border_ceof, border_ceof, border_ceof, cv2.BORDER_CONSTANT,
                              value=col)


def create_tk_root():
    import tkinter as tk
    tk_root = tk._get_default_root()
    if tk_root is None:
        tk_root = tk.Tk()
        tk_root.withdraw()

    return tk_root