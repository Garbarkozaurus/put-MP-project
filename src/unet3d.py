import torch
from torch import nn, cat
from torch.nn.functional import relu
import torch.optim as optim
import torchinfo


from D_shape_reconstruction_dataset import ShapeReconstructionDataset
from custom_loss import CustomLoss

# For testing purposes:
from voxelization import voxelize, create_voxel_from_binary_grid, visualize_voxel_grid
from photo_utils import load_d_image
import numpy as np
import pathlib

_config = {
    "batch_size": 8,
    "learning_rate": 0.01,
    "epochs": 10,
    "optimizer": "Adam",  # Name of optim class as string, e.g. "SGD" or "Adam"
    "momentum": 0.9,      # ignored if optimizer != "SGD"
}

VERBOSE = True


def vprint(*args, **kwargs) -> None:
    if not VERBOSE:
        return
    print(*args, **kwargs)

class UNet(nn.Module):
    # What can still be added: Fully connected layers at bottom of the U, residual connections between each layer, add more layers at each BASE depth
    def __init__(self, device, n_images=30):
        super().__init__()
        self.device = device
        BASE = 4

        # Encoder; 30x64x64x64 -> 4x64x64x64
        self.enc11 = nn.Conv3d(n_images, BASE, kernel_size=3, padding=1, padding_mode='reflect').to(device)
        self.enc12 = nn.Conv3d(BASE, BASE, kernel_size=3, padding=1, padding_mode='reflect').to(device)
        # Downscaling block; 4x64x64x64 -> 8x32x32x32
        self.down1 = nn.Conv3d(BASE, 2*BASE, kernel_size=2, stride=2).to(device)

        self.enc21 = nn.Conv3d(2*BASE, 2*BASE, kernel_size=3, padding=1, padding_mode='reflect').to(device)
        self.enc22 = nn.Conv3d(2*BASE, 2*BASE, kernel_size=3, padding=1, padding_mode='reflect').to(device)
        # Downscaling block; 8x32x32x32 -> 16x16x16x16
        self.down2 = nn.Conv3d(2*BASE, 4*BASE, kernel_size=2, stride=2).to(device)

        self.enc31 = nn.Conv3d(4*BASE, 4*BASE, kernel_size=3, padding=1, padding_mode='reflect').to(device)
        self.enc32 = nn.Conv3d(4*BASE, 4*BASE, kernel_size=3, padding=1, padding_mode='reflect').to(device)
        # Downscaling block; 16x16x16x16 -> 32x8x8x8
        self.down3 = nn.Conv3d(4*BASE, 8*BASE, kernel_size=2, stride=2).to(device)

        self.enc41 = nn.Conv3d(8*BASE, 8*BASE, kernel_size=3, padding=1, padding_mode='reflect').to(device)
        self.enc42 = nn.Conv3d(8*BASE, 8*BASE, kernel_size=3, padding=1, padding_mode='reflect').to(device)

        # Decoder; 32x8x8x8 -> 16x16x16x16
        self.upconv1 = nn.ConvTranspose3d(8*BASE, 4*BASE, kernel_size=2, stride=2).to(device)
        self.dec11 = nn.Conv3d(8*BASE, 4*BASE, kernel_size=3, padding=1, padding_mode='reflect').to(device)
        self.dec12 = nn.Conv3d(4*BASE, 4*BASE, kernel_size=3, padding=1, padding_mode='reflect').to(device)

        # Upscaling block; 16x16x16x16 -> 8x32x32x32
        self.upconv2 = nn.ConvTranspose3d(4*BASE, 2*BASE, kernel_size=2, stride=2).to(device)
        self.dec21 = nn.Conv3d(4*BASE, 2*BASE, kernel_size=3, padding=1, padding_mode='reflect').to(device)
        self.dec22 = nn.Conv3d(2*BASE, 2*BASE, kernel_size=3, padding=1, padding_mode='reflect').to(device)

        # Upscaling block; 8x32x32x32 -> 4x64x64x64
        self.upconv3 = nn.ConvTranspose3d(2*BASE, BASE, kernel_size=2, stride=2).to(device)
        self.dec31 = nn.Conv3d(2*BASE, BASE, kernel_size=3, padding=1, padding_mode='reflect').to(device)
        self.dec32 = nn.Conv3d(BASE, BASE, kernel_size=3, padding=1, padding_mode='reflect').to(device)

        # Output; 4x64x64x64 -> 1x64x64x64
        self.outconv = nn.Conv3d(BASE, 1, kernel_size=3, padding=1, padding_mode='reflect').to(device)
        self.sigmoid = nn.Sigmoid().to(device)

    def forward(self, X):
        print(f"INPUT: X:{X.shape}, X.dtype:{X.dtype}")
        # Encode
        xe11 = relu(self.enc11(X))
        xe12 = relu(self.enc12(xe11))
        xp1 = self.down1(xe12)

        xe21 = relu(self.enc21(xp1))
        xe22 = relu(self.enc22(xe21))
        xp2 = self.down2(xe22)

        xe31 = relu(self.enc31(xp2))
        xe32 = relu(self.enc32(xe31))
        xp3 = self.down3(xe32)

        xe41 = relu(self.enc41(xp3))
        xe42 = relu(self.enc42(xe41))

        # Decode

        xuc1 = self.upconv1(xe42)
        xc1 = cat([xuc1, xe32], dim=1)
        xd11 = relu(self.dec11(xc1))
        xd12 = relu(self.dec12(xd11))

        xuc2 = self.upconv2(xd12)
        xc2 = cat([xuc2, xe22], dim=1)
        xd21 = relu(self.dec21(xc2))
        xd22 = relu(self.dec22(xd21))

        xuc3 = self.upconv3(xd22)
        xc3 = cat([xuc3, xe12], dim=1)
        xd31 = relu(self.dec31(xc3))
        xd32 = relu(self.dec32(xd31))

        # Output
        outconv = self.outconv(xd32)
        out = self.sigmoid(outconv)
        print(f"OUTPUT: out:{out.shape}, out.dtype:{out.dtype}")

        return out


def train(
        inputs_root_dir: str = "RGBD_ModelNet40_centered",
        outputs_root_dir: str = "ModelNet40_ones") -> UNet:
    """Instantiates a new CNN and trains it on the given data.

    The training dataset is formed by concatenating files on the `train_files`
    list.
    """
    train_dataset = ShapeReconstructionDataset(inputs_root_dir, outputs_root_dir, 3, 1, 0, "train")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=_config["batch_size"],
                                               shuffle=False, num_workers=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn = UNet(device, train_dataset.inputs.shape[1])
    vprint(cnn)

    # loss_fn = nn.L1Loss()
    # loss_fn = nn.BCELoss()
    # loss_fn = nn.BCEWithLogitsLoss()
    loss_fn = CustomLoss()

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
        if epoch % 5 == 0 and epoch != 0:
            # Halve learning rate every 5 epochs
            for param_group in optimizer.param_groups:
                param_group["lr"] /= 2
        for i, data in enumerate(train_loader):
            inputs, targets = data
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            output = cnn(inputs)
            if i == 0 and epoch == 0:
                print(f"Output shape: {output.shape}")
                print(f"Targets shape: {targets.shape}")
                print(f"Output min: {torch.min(targets)}, Output mean: {torch.mean(targets)} Output max: {torch.max(targets)}")
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

def test(
        cnn: UNet,
        inputs_root_dir: str = "RGBD_ModelNet40_centered",
        outputs_root_dir: str = "ModelNet40_ones") -> UNet:
    test_dataset = ShapeReconstructionDataset(inputs_root_dir, outputs_root_dir, "test")
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=_config["batch_size"],
                                               shuffle=True, num_workers=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # loss_fn = nn.L1Loss()
    loss_fn = nn.BCELoss()
    # loss_fn = nn.BCEWithLogitsLoss()

    cnn.eval()
    test_loss = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, targets = data
            inputs = inputs.to(device)
            targets = targets.to(device)
            output = cnn(inputs)
            loss = loss_fn(output, targets)
            test_loss += loss
    print(f"Finished testing with loss: {test_loss}")



if __name__ == "__main__":
    cnn = train()
    torch.save(cnn.state_dict(), "saved_model.pt5")
    # test(cnn)
    # cnn = UNet(torch.device("cuda"), 15)
    # cnn.load_state_dict(torch.load("saved_model.pt5"))
    # torchinfo.summary(cnn)
    # Test the model on a given off
    # model_path = "ModelNet40_centered/airplane/test/airplane_0627.off"
    # images_path = "RGBD_ModelNet40_centered/airplane/test/airplane_0627_r_000.png"
    # inputs_dir = pathlib.Path(f"RGBD_ModelNet40_centered/airplane/test")
    # input_file_list = inputs_dir.glob(f"airplane_0627_r_[0-9][0-9][0-9].png")
    # model_path = "ModelNet40_centered/bed/train/bed_0005.off"
    # images_path = "RGBD_ModelNet40_centered/bed/train/bed_0005_r_000.png"
    # inputs_dir = pathlib.Path(f"RGBD_ModelNet40_centered/bed/train")
    # input_file_list = inputs_dir.glob(f"bed_0005_r_[0-9][0-9][0-9].png")
    # d_images = torch.tensor(np.array([load_d_image(str(f)) for f in input_file_list])[::2], dtype=torch.float32).to(torch.device("cuda"))
    # cnn.eval()
    # out = cnn(d_images.reshape(1, 15, 256, 256))
    # out = out.cpu().detach().numpy()
    # out = out.reshape(64, 64, 64)
    # # print(out)
    # print(np.min(out), np.max(out), np.mean(out))
    # out[out >= 0.2] = 1.0
    # out[out < 0.2] = 0
    # print(np.min(out), np.max(out), np.mean(out))
    # print(out.shape)
    # # voxelize(model_path, True, 64)
    # voxel_grid = create_voxel_from_binary_grid(out)
    # visualize_voxel_grid(voxel_grid)
