from unet3d import UNet
import torch
import numpy as np
import pathlib
from voxelization import load_ones_indices_into_binary_grid, voxelize, create_voxel_from_binary_grid, visualize_voxel_grid

cnn = UNet(torch.device("cuda"), 10)
cnn.load_state_dict(torch.load("saved_model.pt5"))
# torchinfo.summary(cnn)
# Test the model on a given off
model_path = "ModelNet40_centered/bench/train/bench_0005.off"
images_path = "ModelNet40_voxel_input/bench/train/bench_0005_r_000.txt"
inputs_dir = pathlib.Path(f"ModelNet40_voxel_input/bench/train")
input_file_list = inputs_dir.glob(f"bench_0005_r_[0-9][0-9][0-9].txt")

inputs = torch.tensor(np.array([load_ones_indices_into_binary_grid(str(f)) for f in input_file_list])[::3], dtype=torch.float32).to(torch.device("cuda"))
cnn.eval()
out = cnn(inputs.reshape(1, 10, 64, 64, 64))
out = out.cpu().detach().numpy()
out = out.reshape(64, 64, 64)
# print(out)
print(np.min(out), np.max(out), np.mean(out))
out[out >= 0.4] = 1.0
out[out < 0.4] = 0.0
print(np.min(out), np.max(out), np.mean(out))
voxelize(model_path, False, 64)
voxel_grid = create_voxel_from_binary_grid(out)
visualize_voxel_grid(voxel_grid)