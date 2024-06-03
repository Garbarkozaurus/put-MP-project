import torch
from torch.utils.data import Dataset
import numpy as np
import pathlib
import os
from typing import Literal
from voxelization import load_ones_indices_into_binary_grid

class ShapeReconstructionDataset(Dataset):
    def __init__(
            self, inputs_root_dir: str = ".\ModelNet40_voxel_input",
            outputs_root_dir: str = ".\ModelNet40_ones",
            every_n: int = 1,
            n_classes: int = 40,
            skip_first_n: int = 0,
            mode: Literal["train"] | Literal["test"] = "train") -> None:
        outputs_dir = pathlib.Path(outputs_root_dir)
        self.inputs = []
        self.output = []
        for i, class_dir in enumerate(outputs_dir.iterdir()):
            if i < skip_first_n:
                continue
            class_name = os.path.basename(class_dir)
            print(f"{i:2}. {class_name}")
            output_file_list = sorted(list(pathlib.Path(f"{class_dir}/{mode}").iterdir()))
            inputs_dir = pathlib.Path(f"{inputs_root_dir}/{class_name}/{mode}")
            for j, output_file in enumerate(output_file_list):
                file_basename = os.path.basename(output_file).rstrip('.txt')
                input_file_list = inputs_dir.glob(f"{file_basename}_r_[0-9][0-9][0-9].txt")
                input_voxels = torch.tensor(np.array([load_ones_indices_into_binary_grid(str(f)) for f in input_file_list][::every_n]), dtype=torch.float32)
                output_voxels = torch.tensor(np.array(load_ones_indices_into_binary_grid(output_file)).reshape(1,64,64,64), dtype=torch.float32)
                if input_voxels.shape != (np.ceil(30.0/every_n), 64, 64, 64):
                    print(input_voxels.shape, output_file)
                self.inputs.append(input_voxels)
                self.output.append(output_voxels)
            if i == n_classes+skip_first_n-1:
                break
        self.inputs = torch.stack(self.inputs)
        self.output = torch.stack(self.output)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (self.inputs[index], self.output[index])

    def __len__(self) -> int:
        return len(self.inputs)


if __name__ == "__main__":
    srd = ShapeReconstructionDataset(mode="train", n_classes=1, skip_first_n=1, every_n=3)
    # print(srd[0])
    print(len(srd))
    print(srd[0][0].shape)
    print(srd[0][1].shape)
