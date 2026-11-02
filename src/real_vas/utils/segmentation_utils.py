import types
from collections import OrderedDict
from tqdm import tqdm
from typing import Optional

from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import torch

from lang_sam.models.gdino import GDINO
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor

from real_vas.utils.frame_utils import preprocess_frames
# from real_vas.utils.sam3_utils import Sam3VideoPredictor
# from real_vas.utils.sam3_utils import Sam3VideoPredictor
from real_vas.utils.vis_utils import add_text, overlay_mask_over_image


@torch.inference_mode()
def init_state(
        self,
        images,
        video_height,
        video_width,
        compute_device,
        offload_video_to_cpu=False,
        offload_state_to_cpu=False,
):
    """Initialize an inference state."""
    inference_state = {}
    inference_state["images"] = images
    inference_state["num_frames"] = len(images)
    # whether to offload the video frames to CPU memory
    # turning on this option saves the GPU memory with only a very small overhead
    inference_state["offload_video_to_cpu"] = offload_video_to_cpu
    # whether to offload the inference state to CPU memory
    # turning on this option saves the GPU memory at the cost of a lower tracking fps
    # (e.g. in a test case of 768x768 model, fps dropped from 27 to 24 when tracking one object
    # and from 24 to 21 when tracking two objects)
    inference_state["offload_state_to_cpu"] = offload_state_to_cpu
    # the original video height and width, used for resizing final output scores
    inference_state["video_height"] = video_height
    inference_state["video_width"] = video_width
    inference_state["device"] = compute_device
    if offload_state_to_cpu:
        inference_state["storage_device"] = torch.device("cpu")
    else:
        inference_state["storage_device"] = compute_device
    # inputs on each frame
    inference_state["point_inputs_per_obj"] = {}
    inference_state["mask_inputs_per_obj"] = {}
    # visual features on a small number of recently visited frames for quick interactions
    inference_state["cached_features"] = {}
    # values that don't change across frames (so we only need to hold one copy of them)
    inference_state["constants"] = {}
    # mapping between client-side object id and model-side object index
    inference_state["obj_id_to_idx"] = OrderedDict()
    inference_state["obj_idx_to_id"] = OrderedDict()
    inference_state["obj_ids"] = []
    # A storage to hold the model's tracking results and states on each frame
    inference_state["output_dict"] = {
        "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
        "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
    }
    # Slice (view) of each object tracking results, sharing the same memory with "output_dict"
    inference_state["output_dict_per_obj"] = {}
    # A temporary storage to hold new outputs when user interact with a frame
    # to add clicks or mask (it's merged into "output_dict" before propagation starts)
    inference_state["temp_output_dict_per_obj"] = {}
    # Frames that already holds consolidated outputs from click or mask inputs
    # (we directly use their consolidated outputs during tracking)
    inference_state["consolidated_frame_inds"] = {
        "cond_frame_outputs": set(),  # set containing frame indices
        "non_cond_frame_outputs": set(),  # set containing frame indices
    }
    # metadata for each tracking frame (e.g. which direction it's tracked)
    inference_state["tracking_has_started"] = False
    inference_state["frames_already_tracked"] = {}
    inference_state["frames_tracked_per_obj"] = {}

    # Warm up the visual backbone and cache the image feature on frame 0
    self._get_image_feature(inference_state, frame_idx=0, batch_size=1)

    return inference_state


def get_sam2_predictor(single_image=False, model_cfg="sam2_hiera_l.yaml", checkpoint="sam2_hiera_large.pt",
                       device="cuda"):
    if single_image:
        return SAM2ImagePredictor(build_sam2(model_cfg, checkpoint, device))

    pred = build_sam2_video_predictor(model_cfg, checkpoint, device)
    # Monkey patch the new init state method to allow for frames to just be fed to the method
    pred.init_state = types.MethodType(init_state, pred)

    return pred


class LangSamRunner:
    def __init__(self, image_sam: SAM2ImagePredictor):
        self.gdino = GDINO()
        self.gdino.build_model()
        self.gdino.model = self.gdino.model.to(image_sam.device)
        self.image_sam = image_sam
        self.moving_objects = [
            "person", "bird", "fish", "insect", "dog", "cat", "horse", "snake", "animal",
            "car", "bike", "motorcycle", "train", "airplane", "boat", "ship", "helicopter", "submarine", "rocket",
            "bus", "truck", "robot", "drone", "conveyor belt", "wind turbine", "fan", "clock hands", "gears",
            "ball", "frisbee", "pendulum", "swing", "yo-yo", "kite", "shopping cart"
        ]
        cmap = plt.get_cmap("tab20")
        self.colours = [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b, _ in (cmap(i) for i in range(cmap.N))]
        self.box_threshold = 0.15
        self.text_threshold = 0.15
        # self.box_threshold = 0.05
        # self.text_threshold = 0.05

    def _run_gdino(self, images: torch.Tensor, text_prompt: list):
        inputs = self.gdino.processor(images=images, text=text_prompt, return_tensors="pt").to(images.device)
        with torch.no_grad():
            outputs = self.gdino.model(**inputs)

        gdino_results = self.gdino.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=torch.stack([torch.tensor([images.shape[1], images.shape[2]])] * images.shape[0]),
        )

        return gdino_results

    def run_custom(self, images: torch.Tensor, prompt: Optional[list[str]] = None) -> \
            list[dict]:
        if prompt is None:
            prompt = self.moving_objects
        text_prompt = ". ".join(prompt) + "."

        gdino_results = self._run_gdino(images, [text_prompt] * len(images))
        # This isn't batched due to complexity with trying to batch in SAM, as they just use a for loop anyways
        final_results = []
        for i, gdino_result in enumerate(gdino_results):
            if not len(gdino_result["boxes"]):
                continue
            sam_results = self._run_sam(gdino_result["boxes"], images[i:i + 1])
            final_results.append(gdino_result)
            final_results[-1]["masks"] = sam_results[0]
            final_results[-1]["mask_scores"] = sam_results[1]

            assert len(final_results[-1]["masks"]) == len(gdino_result["boxes"])

        return final_results

    def _run_sam(self, bboxes: torch.Tensor, images: torch.Tensor):
        self.image_sam.reset_predictor()
        self.image_sam._orig_hw = [images.shape[1:3]] * len(images)
        self.image_sam._is_batch = False
        input_image = self.image_sam._transforms.transforms(images.permute(0, 3, 1, 2) / 255.)
        assert (
                len(input_image.shape) == 4 and input_image.shape[1] == 3
        ), f"input_image must be of size 1x3xHxW, got {input_image.shape}"
        backbone_out = self.image_sam.model.forward_image(input_image)
        _, vision_feats, _, _ = self.image_sam.model._prepare_backbone_features(backbone_out)
        # Add no_mem_embed, which is added to the lowest rest feat. map during training on videos
        if self.image_sam.model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.image_sam.model.no_mem_embed
        feats = [feat.permute(1, 2, 0).view(1, -1, *feat_size) for feat, feat_size in
                 zip(vision_feats[::-1], self.image_sam._bb_feat_sizes[::-1])][::-1]
        self.image_sam._features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
        self.image_sam._is_image_set = True
        prep_bboxes = self.image_sam._prep_prompts(None, None, bboxes, None, True)[-1]

        return self.image_sam._predict(None, None, boxes=prep_bboxes, multimask_output=False)

    def visualise(self, frame: Image.Image, single_result: dict):
        mask_img = np.array(frame).copy()
        for i, (mask, lbl) in enumerate(zip(single_result["masks"], single_result["labels"])):
            if mask.ndim == 3:
                mask = mask[0]
            mask = mask.cpu().numpy()
            mask_img = overlay_mask_over_image(mask_img, mask, alpha=0.5, color=self.colours[i % len(self.colours)])
            top_left = np.argwhere(mask).min(axis=0)
            mask_img = add_text(mask_img, f"{lbl}", (top_left[1], top_left[0] - 30), scale=1.5, thickness=1,
                                bg_col=(255, 255, 255))

        return Image.fromarray(mask_img)

    @staticmethod
    def filter_results_by_idxs(results, idxs):
        return {
            k: v[idxs] if isinstance(v, torch.Tensor) else [results["labels"][idx] for idx in idxs]
            for k, v in results.items()
        }

    def get_top_k_masks_from_result(self, single_result: dict, top_k=10):
        top_idxs = torch.argsort(single_result["scores"], descending=True)[:top_k]

        return self.filter_results_by_idxs(single_result, top_idxs)

    @staticmethod
    def get_masks(results: list[dict]):
        return [r["masks"] for r in results]

    @staticmethod
    def get_labels(results: list[dict]):
        return [r["labels"] for r in results]

    @staticmethod
    def get_scores(results: list[dict]):
        return [r["scores"] for r in results]

    def remove_from_device(self):
        self.gdino.model = self.gdino.model.to("cpu")
        self.image_sam.model = self.image_sam.model.to("cpu")

    def to_device(self, device):
        self.gdino.model = self.gdino.model.to(device)
        self.image_sam.model = self.image_sam.model.to(device)


class SAMVid(torch.nn.Module):
    img_mean = (0.485, 0.456, 0.406)
    img_std = (0.229, 0.224, 0.225)

    def __init__(self, ckpt, device="cpu"):
        super().__init__()
        self.pipeline = get_sam2_predictor(single_image=False, checkpoint=ckpt, device=device)

    # TODO this is repeated in a couple of places so just need a single method for it
    def forward(self, vid_frames, first_frame_masks, device="cuda", resize_to=None, already_preprocessed=False):
        if first_frame_masks.ndim == 2:
            first_frame_masks = first_frame_masks[None]
        if first_frame_masks.ndim == 3:
            first_frame_masks = first_frame_masks[:, None]
        num_input_masks = first_frame_masks.shape[0]
        if num_input_masks == 0:
            end_shape = resize_to if resize_to is not None else (self.pipeline.image_size, self.pipeline.image_size)
            return torch.empty(0, vid_frames.shape[0], *end_shape, dtype=torch.bool)
        if not already_preprocessed:
            vid_frames = preprocess_frames(vid_frames, img_mean=self.img_mean, img_std=self.img_std, out_max=1,
                                           resize_to=(self.pipeline.image_size, self.pipeline.image_size), device=None)
        vid_frames = vid_frames.to(device)
        self.pipeline = self.pipeline.to(device)
        inference_state = self.pipeline.init_state(vid_frames, self.pipeline.image_size,
                                                   self.pipeline.image_size, device)
        for mask_idx, mask in enumerate(tqdm(first_frame_masks, desc="Tracking masks")):
            _, _, out_mask_logits = self.pipeline.add_new_mask(inference_state, 0, mask_idx + 1, mask[0])
        sam_vid_masks = torch.stack([(out_mask_logits[:, 0] > 0.0) for _, _, out_mask_logits in
                                     self.pipeline.propagate_in_video(inference_state)])

        if resize_to is not None:
            sam_vid_masks = torch.nn.functional.interpolate(sam_vid_masks.to(torch.uint8), resize_to,
                                                            mode="nearest").bool()
        return sam_vid_masks


class SAMImg(torch.nn.Module):
    img_mean = (0.485, 0.456, 0.406)
    img_std = (0.229, 0.224, 0.225)

    def __init__(self, ckpt, device="cpu"):
        super().__init__()
        self.pipeline = get_sam2_predictor(single_image=True, checkpoint=ckpt, device=device)

    # TODO this is repeated in a couple of places so just need a single method for it
    # TODO is there anyway this can stay torch instead of being numpy???
    def forward(self, frame: np.ndarray, positive_points: list | None = None, negative_points: list | None = None,
                bbox: list | None = None):
        # Prepare image for SAM model
        # TODO this should be handled in pure torch
        self.pipeline.set_image(frame)

        assert positive_points is not None or bbox is not None, "Either positive points or box must be provided for SAMImg."

        if positive_points is not None:
            # Convert points to NumPy array
            positive_points_np = np.array(positive_points)
            negative_points_np = np.array(negative_points)

            # Create input labels for SAM: 1 for positive, 0 for negative
            input_labels = np.array([1] * len(positive_points_np) + [0] * len(negative_points_np))

            # Concatenate positive and negative points
            if len(negative_points_np):
                input_points = np.concatenate([positive_points_np, negative_points_np], axis=0)
            else:
                input_points = positive_points_np
            bbox = None
        else:
            # If using bbox, set input points and labels to None
            input_points = None
            input_labels = None

        # Perform prediction
        masks, scores, logits = self.pipeline.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=True,
            box=bbox
        )
        mask = masks[np.argmax(scores)]

        return mask.astype(np.uint8) * 255


def binary_erosion_torch(mask: torch.BoolTensor, kernel_size=3, iterations=1):
    """
    Erodes a binary mask using boolean logic, avoiding float conversion.

    Args:
        mask: Bool tensor of shape (B, 1, H, W)
        kernel_size: Size of erosion kernel (must be odd)
        iterations: Number of erosion passes

    Returns:
        Eroded bool tensor (same shape)
    """
    pad = kernel_size // 2

    for _ in range(iterations):
        mask = ~torch.nn.functional.max_pool2d(
            ~mask,  # invert: True->False, False->True
            kernel_size,
            stride=1,
            padding=pad
        )
    return mask
