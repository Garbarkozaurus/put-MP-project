import torch
from torch.utils.data import Dataset
import numpy as np
import pathlib
import os
from typing import Literal
from voxelization import load_ones_indices_into_binary_grid
from photo_utils import load_rgbd_image


class FBFShapeReconstructionDataset(Dataset):
    def __init__(
            self, inputs_root_dir: str = "RGBD_ModelNet40_centered",
            outputs_root_dir: str = "ModelNet40_ones",
            mode: Literal["train"] | Literal["test"] = "train") -> None:
        self.mode = mode
        self.inputs_root_dir = inputs_root_dir
        self.outputs_root_dir = outputs_root_dir
        outputs_dir = pathlib.Path(outputs_root_dir)
        self.filename_index = []
        for i, class_dir in enumerate(outputs_dir.iterdir()):
            output_file_list = sorted(list(pathlib.Path(f"{class_dir}/{mode}").iterdir()))
            for output_file in output_file_list:
                self.filename_index.append(str(output_file).rstrip(".txt"))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        class_name = self.filename_index[index].split("/")[1]
        file_basename = self.filename_index[index].split("/")[-1]
        inputs_dir = pathlib.Path(f"{self.inputs_root_dir}/{class_name}/{self.mode}/")
        input_file_list = inputs_dir.glob(f"{file_basename}_r_[0-9][0-9][0-9].png")
        rgbd_images = torch.tensor(np.array([load_rgbd_image(str(f)) for f in input_file_list]), dtype=torch.uint8)
        output_file = f"{self.outputs_root_dir}/{class_name}/{self.mode}/{file_basename}.txt"
        binary_grid = torch.tensor(load_ones_indices_into_binary_grid(output_file), dtype=torch.float16)
        return rgbd_images, binary_grid

    def __len__(self) -> int:
        return len(self.filename_index)


if __name__ == "__main__":
    srd = FBFShapeReconstructionDataset(mode="train")
    print(srd[0])
    print(len(srd))
