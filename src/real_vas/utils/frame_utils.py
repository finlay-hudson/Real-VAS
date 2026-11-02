import numpy as np
from PIL import Image
import torch
from torchvision.transforms.functional import pil_to_tensor


def get_resize_dims(height, width, longest_edge: int = 600):
    if height > width:
        # Vertical image
        ratio = longest_edge / height
        new_height = longest_edge
        new_width = int(width * ratio)
    else:
        # Horizontal or square image
        ratio = longest_edge / width
        new_width = longest_edge
        new_height = int(height * ratio)

    return new_height, new_width


def preprocess_frames(frames, img_mean=None, img_std=None, out_max=255, resize_to=None, resize_mode='bilinear',
                      device=None):
    torch_frames = []
    for frame in frames:
        torch_frame = frame
        if isinstance(frame, Image.Image):
            torch_frame = pil_to_tensor(frame)
        if isinstance(frame, np.ndarray):
            torch_frame = torch.from_numpy(frame.copy())
        if not isinstance(torch_frame, torch.Tensor):
            raise ValueError(f"Unsupported frame type: {type(frame)}. Must be torch tensor, PIL Image or numpy array.")
        if torch_frame.shape[-1] == 3:
            torch_frame = torch_frame.permute(2, 0, 1)  # Convert to (C, H, W) format
        if resize_to is not None:
            torch_frame = torch.nn.functional.interpolate(torch_frame[None], size=resize_to, mode=resize_mode,
                                                          align_corners=False)[0]
        torch_frames.append(torch_frame)
    torch_frames = torch.stack(torch_frames)

    if out_max == 1:
        torch_frames = torch_frames / 255.0
    if img_mean is not None:
        # normalize by mean
        torch_frames -= torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
    if img_std is not None:
        # normalize by std
        torch_frames /= torch.tensor(img_std, dtype=torch.float32)[:, None, None]
    if device is not None:
        torch_frames = torch_frames.to(device)
    return torch_frames
