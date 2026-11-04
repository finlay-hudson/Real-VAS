from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from natsort import natsorted
from torch.utils.data import Dataset
from torchvision import transforms


@dataclass
class DataConfigREALVAS:
    data_root: Path = Path("data/Real-VAS")
    frame_dir_name: Optional[str] = "frames"
    amodal_mask_dir_name: str = "amodal_masks"
    modal_mask_dir_name: str = "modal_masks"
    anno_file_name: Optional[str] = "annos"


class DatasetSplit(Enum):
    CONTAINMENT = "CONTAINMENT"
    OCCLUSION = "OCCLUSION"
    BOTH = "BOTH"


class REALVASDataset(Dataset):
    def __init__(self, dataset_cfg: DataConfigREALVAS, dataset_split: DatasetSplit = DatasetSplit.BOTH):
        self.ds_root = dataset_cfg.data_root
        assert self.ds_root.exists(), f"Dataset root: {self.ds_root} does not exists"
        self.frame_dir_name = dataset_cfg.frame_dir_name
        self.amodal_mask_dir_name = dataset_cfg.amodal_mask_dir_name
        self.modal_mask_dir_name = dataset_cfg.modal_mask_dir_name
        self.anno_file_name = dataset_cfg.anno_file_name
        self.dataset_split = dataset_split
        self.vid_dirs = []
        if self.dataset_split in [DatasetSplit.BOTH, DatasetSplit.CONTAINMENT]:
            self.vid_dirs.extend(self._get_dirs("containment"))
        if self.dataset_split in [DatasetSplit.BOTH, DatasetSplit.OCCLUSION]:
            self.vid_dirs.extend(self._get_dirs("occlusion"))
        self.transform = transforms.ToTensor()

    def _get_dirs(self, split: str):
        return [d.relative_to(self.ds_root) for d in (self.ds_root / split).glob('*') if d.is_dir()]

    def _get_ims(self, dir: Path, suffix: str = ".png") -> list[torch.Tensor]:
        return torch.stack([self.transform(Image.open(m)) for m in natsorted(list((dir).glob(f"*{suffix}")))])

    def get_data_for_vid(self, vid_name: Path) -> (
            torch.Tensor, torch.Tensor, Optional[list[torch.Tensor]], list[torch.Tensor], Optional[list]):
        vid_dir = self.ds_root / vid_name
        anno = None
        if self.anno_file_name:
            with open(vid_dir / (self.anno_file_name + ".json")) as f:
                anno = json.load(f)
        frames = None
        if self.frame_dir_name:
            frames = self._get_ims(vid_dir / self.frame_dir_name, suffix=".jpg")

        amodal_masks = None
        if self.amodal_mask_dir_name:
            amodal_masks = self._get_ims(vid_dir / self.amodal_mask_dir_name, suffix=".png")
        modal_masks = self._get_ims(vid_dir / self.modal_mask_dir_name, suffix=".png")

        first_amodal = amodal_masks[0] if amodal_masks is not None else None
        return frames, first_amodal, amodal_masks, modal_masks, anno, str(vid_name)

    def __len__(self):
        return len(self.vid_dirs)

    def __getitem__(self, idx):
        return self.get_data_for_vid(self.vid_dirs[idx])
