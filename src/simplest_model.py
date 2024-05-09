import torch
import torch.nn as nn
import torch.optim as optim
import torchinfo


from shape_reconstruction_dataset import ShapeReconstructionDataset
# from fbf_shape_reconstruction_dataset import FBFShapeReconstructionDataset


_config = {
    "batch_size": 8,
    "learning_rate": 0.001,
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
            nn.Conv3d(30, 30, kernel_size=3, padding=1).to(device),
            nn.ReLU(),
            # setting stride like this makes the network reduce the size
            # of "each image" in half, but preserve all channels
            nn.Conv3d(30, 32, kernel_size=3, padding=1, stride=(2, 2, 1)).to(device),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, padding=1, stride=(2, 2, 1)).to(device),
            nn.ReLU(),
            # reduce the 4 channels to 1 with the next two layers
            nn.Conv3d(64, 64, kernel_size=3, padding=1, stride=(1, 1, 2)).to(device),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1, stride=(1, 1, 2)).to(device),
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
        for i, data in enumerate(train_loader):
            inputs, targets = data
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            output = cnn(inputs)[:, :, :, :, 0]
            loss = loss_fn(output, targets)
            net_state = str(cnn.state_dict())
            loss.backward()
            epoch_loss += loss
            optimizer.step()
            if str(cnn.state_dict()) == net_state:
                vprint("Network not updating")
        print(f"Finished epoch {epoch+1} with loss: {epoch_loss}")
    print("Finished training CnnBasic")
    # torch.save(cnn.state_dict(), "saved_model.pt5")
    return cnn


if __name__ == "__main__":
    cnn = train()
    torchinfo.summary(cnn)
