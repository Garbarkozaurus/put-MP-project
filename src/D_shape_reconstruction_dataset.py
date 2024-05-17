import torch
from torch.utils.data import Dataset
import numpy as np
import pathlib
import os
from typing import Literal
from voxelization import load_ones_indices_into_binary_grid
from photo_utils import load_d_image


class ShapeReconstructionDataset(Dataset):
    def __init__(
            self, inputs_root_dir: str = "RGBD_ModelNet40_centered",
            outputs_root_dir: str = "ModelNet40_ones",
            mode: Literal["train"] | Literal["test"] = "train") -> None:
        outputs_dir = pathlib.Path(outputs_root_dir)
        self.images = []
        self.voxels = []
        for i, class_dir in enumerate(outputs_dir.iterdir()):
            class_name = os.path.basename(class_dir)
            print(f"{i:2}. {class_name}")
            output_file_list = sorted(list(pathlib.Path(f"{class_dir}/{mode}").iterdir()))
            inputs_dir = pathlib.Path(f"{inputs_root_dir}/{class_name}/{mode}")
            for output_file in output_file_list:
                # find all pictures associated with the voxel
                file_basename = os.path.basename(output_file).rstrip('.txt')
                input_file_list = inputs_dir.glob(f"{file_basename}_r_[0-9][0-9][0-9].png")
                # [:, :, -2] takes only b and d channels
                rgbd_images = torch.tensor(np.array([load_d_image(str(f)) for f in input_file_list][::3]), dtype=torch.uint8)
                binary_grid = torch.tensor(load_ones_indices_into_binary_grid(output_file), dtype=torch.float16)
                # self.data.append((rgbd_images, binary_grid))
                if rgbd_images.shape != (10, 256, 256):
                    print(rgbd_images.shape, output_file)
                self.images.append(rgbd_images)
                self.voxels.append(binary_grid)
            if i == 20:
                break
        self.images = np.array(self.images, dtype=np.float32) / 255.0
        self.voxels = np.array(self.voxels, dtype=np.uint8)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (self.images[index], self.voxels[index])

    def __len__(self) -> int:
        return len(self.images)


if __name__ == "__main__":
    srd = ShapeReconstructionDataset(mode="train")
    print(srd[0])
    print(len(srd))
