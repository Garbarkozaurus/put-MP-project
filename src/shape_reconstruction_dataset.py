import torch
from torch.utils.data import Dataset
import numpy as np
import pathlib
import os
from typing import Literal
from voxelization import load_ones_indices_into_binary_grid
from photo_utils import load_rgbd_image


class ShapeReconstructionDataset(Dataset):
    def __init__(
            self, inputs_root_dir: str = "RGBD_ModelNet40_centered",
            outputs_root_dir: str = "ModelNet40_ones",
            mode: Literal["train"] | Literal["test"] = "train") -> None:
        outputs_dir = pathlib.Path(outputs_root_dir)
        self.data = []
        for i, class_dir in enumerate(outputs_dir.iterdir()):
            class_name = os.path.basename(class_dir)
            print(f"{i:2}. {class_name}")
            output_file_list = sorted(list(pathlib.Path(f"{class_dir}/{mode}").iterdir()))
            inputs_dir = pathlib.Path(f"{inputs_root_dir}/{class_name}/{mode}")
            for output_file in output_file_list:
                # find all pictures associated with the voxel
                file_basename = os.path.basename(output_file).rstrip('.txt')
                input_file_list = inputs_dir.glob(f"{file_basename}_r_[0-9][0-9][0-9].png")
                rgbd_images = torch.tensor(np.array([load_rgbd_image(str(f)) for f in input_file_list]), dtype=torch.uint8)
                binary_grid = torch.tensor(load_ones_indices_into_binary_grid(output_file), dtype=torch.float16)
                self.data.append((rgbd_images, binary_grid))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)


if __name__ == "__main__":
    srd = ShapeReconstructionDataset(mode="test")
    print(srd[0])
    print(len(srd))
