import torch
import torch.nn as nn
import torch.optim as optim
import torchinfo


from D_shape_reconstruction_dataset import ShapeReconstructionDataset
# from fbf_shape_reconstruction_dataset import FBFShapeReconstructionDataset

# For testing purposes:
from voxelization import voxelize, create_voxel_from_binary_grid, visualize_voxel_grid
from photo_utils import load_d_image
import numpy as np
import pathlib

_config = {
    "batch_size": 8,
    "learning_rate": 0.1,
    "epochs": 10,
    "optimizer": "Adam",  # Name of optim class as string, e.g. "SGD" or "Adam"
    "momentum": 0.9,      # ignored if optimizer != "SGD"
}

VERBOSE = True


def vprint(*args, **kwargs) -> None:
    if not VERBOSE:
        return
    print(*args, **kwargs)


class CnnBasic(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.convolutional_block = nn.Sequential(
            nn.Conv2d(15, 30, kernel_size=3, padding=1).to(device),
            nn.ReLU(),
            # setting stride like this makes the network reduce the size
            # of "each image" in half, but preserve all channels
            nn.Conv2d(30, 32, kernel_size=3, padding=1, stride=2).to(device),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2).to(device),
            nn.ReLU(),
            # reduce the 4 channels to 1 with the next two layers
            nn.Conv2d(64, 64, kernel_size=3, padding=1).to(device),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1).to(device),
            nn.ReLU(),
        )
        self.out_activation = nn.Sequential(
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.float()
        x = self.convolutional_block(x)
        x = self.out_activation(x)
        return x


def train(
        inputs_root_dir: str = "RGBD_ModelNet40_centered",
        outputs_root_dir: str = "ModelNet40_ones") -> CnnBasic:
    """Instantiates a new CNN and trains it on the given data.

    The training dataset is formed by concatenating files on the `train_files`
    list.
    """
    train_dataset = ShapeReconstructionDataset(inputs_root_dir, outputs_root_dir, "train")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=_config["batch_size"],
                                               shuffle=True, num_workers=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn = CnnBasic(device)
    vprint(cnn)

    loss_fn = nn.L1Loss()

    # Construct optimizer kwargs
    optimizer_cls = optim.__getattribute__(_config["optimizer"])
    assert issubclass(optimizer_cls, optim.Optimizer), "optimizer is not a valid class name"
    optimizer_kwargs = {}
    optimizer_kwargs.update({"lr": _config["learning_rate"]})
    match optimizer_cls:
        case optim.SGD:
            if "momentum" in _config:
                optimizer_kwargs.update({"momentum": _config["momentum"]})

    optimizer = optimizer_cls(cnn.parameters(), **optimizer_kwargs)
    for epoch in range(_config["epochs"]):
        print(f"Starting epoch {epoch+1}/{_config['epochs']}")
        epoch_loss = 0
        if epoch % 2 == 1:
            # Decrease learning rate every 2 epochs
            for param_group in optimizer.param_groups:
                param_group["lr"] /= 2
        for i, data in enumerate(train_loader):
            inputs, targets = data
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            if epoch == 0 and i == 0:
                print(inputs.shape, targets.shape)
                print(inputs.dtype, targets.dtype)
                print(inputs[0, 0, 0, :5], targets[0, 0, 0, :5])
            output = cnn(inputs)
            loss = loss_fn(output, targets)
            # net_state = str(cnn.state_dict())
            loss.backward()
            epoch_loss += loss
            optimizer.step()
            # if str(cnn.state_dict()) == net_state:
            #     vprint("Network not updating")
        print(f"Finished epoch {epoch+1} with loss: {epoch_loss}")
    print("Finished training CnnBasic")
    # torch.save(cnn.state_dict(), "saved_model.pt5")
    return cnn


if __name__ == "__main__":
    cnn = train()
    # cnn = CnnBasic(torch.device("cpu"))
    # cnn.load_state_dict(torch.load("saved_model.pt5"))
    torchinfo.summary(cnn)
    # torch.save(cnn.state_dict(), "saved_model.pt5")
    # # Test the model on a given off
    # model_path = "ModelNet40_centered/airplane/test/airplane_0627.off"
    # images_path = "RGBD_ModelNet40_centered/airplane/test/airplane_0627_r_000.png"
    # inputs_dir = pathlib.Path(f"RGBD_ModelNet40_centered/airplane/test")
    # input_file_list = inputs_dir.glob(f"airplane_0627_r_[0-9][0-9][0-9].png")
    # d_images = torch.tensor(np.array([load_d_image(str(f)) for f in input_file_list][::2]), dtype=torch.uint8)
    # cnn.eval()
    # out = cnn(d_images.reshape(1, 15, 256, 256))
    # out = np.array(out.detach().numpy())[0]
    # print(out)
    # print(np.min(out), np.max(out), np.mean(out))
    # out[out >= 0.5] = 1
    # out[out < 0.5] = 0
    # print(np.min(out), np.max(out), np.mean(out))
    # print(out.shape)
    # voxelize(model_path, True, 64)
    # voxel_grid = create_voxel_from_binary_grid(out)
    # visualize_voxel_grid(voxel_grid)
