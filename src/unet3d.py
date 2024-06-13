import torch
from torch import nn, cat
from torch.nn.functional import relu
import torch.optim as optim
import torchinfo


from D_shape_reconstruction_dataset import ShapeReconstructionDataset
from custom_loss import CustomLoss

_config = {
    "batch_size": 8,
    "learning_rate": 0.001,
    "epochs": 100,
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
        BASE = 16
        self.BASE = BASE
        # Input; 1x64x64x64 -> 4x64x64x64
        self.inconv = nn.Conv3d(n_images, BASE, kernel_size=3, padding=1, padding_mode='reflect').to(device)

        # Encoder; 30x64x64x64 -> 4x64x64x64
        self.enc11 = nn.Conv3d(BASE, BASE, kernel_size=3, padding=1, padding_mode='reflect').to(device)
        self.enc12 = nn.Conv3d(BASE, BASE, kernel_size=3, padding=1, padding_mode='reflect').to(device)
        self.enc13 = nn.Conv3d(BASE, BASE, kernel_size=3, padding=1, padding_mode='reflect').to(device)
        self.enc14 = nn.Conv3d(BASE, BASE, kernel_size=3, padding=1, padding_mode='reflect').to(device)
        # Downscaling block; 4x64x64x64 -> 8x32x32x32
        self.down1 = nn.Conv3d(BASE, 2*BASE, kernel_size=2, stride=2).to(device)

        self.enc21 = nn.Conv3d(2*BASE, 2*BASE, kernel_size=3, padding=1, padding_mode='reflect').to(device)
        self.enc22 = nn.Conv3d(2*BASE, 2*BASE, kernel_size=3, padding=1, padding_mode='reflect').to(device)
        self.enc23 = nn.Conv3d(2*BASE, 2*BASE, kernel_size=3, padding=1, padding_mode='reflect').to(device)
        self.enc24 = nn.Conv3d(2*BASE, 2*BASE, kernel_size=3, padding=1, padding_mode='reflect').to(device)
        # Downscaling block; 8x32x32x32 -> 16x16x16x16
        self.down2 = nn.Conv3d(2*BASE, 4*BASE, kernel_size=2, stride=2).to(device)

        self.enc31 = nn.Conv3d(4*BASE, 4*BASE, kernel_size=3, padding=1, padding_mode='reflect').to(device)
        self.enc32 = nn.Conv3d(4*BASE, 4*BASE, kernel_size=3, padding=1, padding_mode='reflect').to(device)
        self.enc33 = nn.Conv3d(4*BASE, 4*BASE, kernel_size=3, padding=1, padding_mode='reflect').to(device)
        self.enc34 = nn.Conv3d(4*BASE, 4*BASE, kernel_size=3, padding=1, padding_mode='reflect').to(device)
        # Downscaling block; 16x16x16x16 -> 32x8x8x8
        self.down3 = nn.Conv3d(4*BASE, 8*BASE, kernel_size=2, stride=2).to(device)

        self.enc41 = nn.Conv3d(8*BASE, 8*BASE, kernel_size=3, padding=1, padding_mode='reflect').to(device)
        self.enc42 = nn.Conv3d(8*BASE, 8*BASE, kernel_size=3, padding=1, padding_mode='reflect').to(device)
        self.enc43 = nn.Conv3d(8*BASE, 8*BASE, kernel_size=3, padding=1, padding_mode='reflect').to(device)
        self.enc44 = nn.Conv3d(8*BASE, 8*BASE, kernel_size=3, padding=1, padding_mode='reflect').to(device)

        self.fc1 = nn.Linear(8*BASE*8*8*8, 500).to(device)
        self.fc2 = nn.Linear(500, 8*BASE*8*8*8).to(device)

        # Decoder; 32x8x8x8 -> 16x16x16x16
        self.upconv1 = nn.ConvTranspose3d(8*BASE, 4*BASE, kernel_size=2, stride=2).to(device)
        self.dec11 = nn.Conv3d(8*BASE, 4*BASE, kernel_size=3, padding=1, padding_mode='reflect').to(device)
        self.dec12 = nn.Conv3d(4*BASE, 4*BASE, kernel_size=3, padding=1, padding_mode='reflect').to(device)
        self.dec13 = nn.Conv3d(4*BASE, 4*BASE, kernel_size=3, padding=1, padding_mode='reflect').to(device)
        self.dec14 = nn.Conv3d(4*BASE, 4*BASE, kernel_size=3, padding=1, padding_mode='reflect').to(device)

        # Upscaling block; 16x16x16x16 -> 8x32x32x32
        self.upconv2 = nn.ConvTranspose3d(4*BASE, 2*BASE, kernel_size=2, stride=2).to(device)
        self.dec21 = nn.Conv3d(4*BASE, 2*BASE, kernel_size=3, padding=1, padding_mode='reflect').to(device)
        self.dec22 = nn.Conv3d(2*BASE, 2*BASE, kernel_size=3, padding=1, padding_mode='reflect').to(device)
        self.dec23 = nn.Conv3d(2*BASE, 2*BASE, kernel_size=3, padding=1, padding_mode='reflect').to(device)
        self.dec24 = nn.Conv3d(2*BASE, 2*BASE, kernel_size=3, padding=1, padding_mode='reflect').to(device)

        # Upscaling block; 8x32x32x32 -> 4x64x64x64
        self.upconv3 = nn.ConvTranspose3d(2*BASE, BASE, kernel_size=2, stride=2).to(device)
        self.dec31 = nn.Conv3d(2*BASE, BASE, kernel_size=3, padding=1, padding_mode='reflect').to(device)
        self.dec32 = nn.Conv3d(BASE, BASE, kernel_size=3, padding=1, padding_mode='reflect').to(device)
        self.dec33 = nn.Conv3d(BASE, BASE, kernel_size=3, padding=1, padding_mode='reflect').to(device)
        self.dec34 = nn.Conv3d(BASE, BASE, kernel_size=3, padding=1, padding_mode='reflect').to(device)

        # Output; 4x64x64x64 -> 1x64x64x64
        self.outconv = nn.Conv3d(BASE, 1, kernel_size=3, padding=1, padding_mode='reflect').to(device)
        self.sigmoid = nn.Sigmoid().to(device)

    def forward(self, X):
        # print(f"INPUT: X:{X.shape}, X.dtype:{X.dtype}")
        # Input
        xin = relu(self.inconv(X))
        res = xin
        # Encode
        xe11 = relu(self.enc11(xin))
        xe12 = relu(self.enc12(xe11))
        xe12 = relu(xe12+res)
        res = xe12
        xe13 = relu(self.enc13(xe12))
        xe14 = relu(self.enc14(xe13))
        xe14 = relu(xe14+res)
        xp1 = self.down1(xe14)

        res = xp1
        xe21 = relu(self.enc21(xp1))
        xe22 = relu(self.enc22(xe21))
        xe22 = relu(xe22+res)
        res = xe22
        xe23 = relu(self.enc23(xe22))
        xe24 = relu(self.enc24(xe23))
        xe24 = relu(xe24+res)
        xp2 = self.down2(xe24)

        res = xp2
        xe31 = relu(self.enc31(xp2))
        xe32 = relu(self.enc32(xe31))
        xe32 = relu(xe32+res)
        res = xe32
        xe33 = relu(self.enc33(xe32))
        xe34 = relu(self.enc34(xe33))
        xe34 = relu(xe34+res)
        xp3 = self.down3(xe34)

        res = xp3
        xe41 = relu(self.enc41(xp3))
        xe42 = relu(self.enc42(xe41))
        xe42 = relu(xe42+res)
        res = xe42
        xe43 = relu(self.enc43(xe42))
        xe44 = relu(self.enc44(xe43))
        xe44 = relu(xe44+res)

        # Fully connected
        x = xe44.view(-1, 8*self.BASE*8*8*8)
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        x = x.view(-1, 8*self.BASE, 8, 8, 8)

        # Decode

        xuc1 = self.upconv1(x)
        res = xuc1
        xc1 = cat([xuc1, xe32], dim=1)
        xd11 = relu(self.dec11(xc1))
        xd12 = relu(self.dec12(xd11))
        xd12 = relu(xd12+res)
        res = xd12
        xd13 = relu(self.dec13(xd12))
        xd14 = relu(self.dec14(xd13))
        xd14 = relu(xd14+res)

        xuc2 = self.upconv2(xd14)
        res = xuc2
        xc2 = cat([xuc2, xe22], dim=1)
        xd21 = relu(self.dec21(xc2))
        xd22 = relu(self.dec22(xd21))
        xd22 = relu(xd22+res)
        res = xd22
        xd23 = relu(self.dec23(xd22))
        xd24 = relu(self.dec24(xd23))
        xd24 = relu(xd24+res)

        xuc3 = self.upconv3(xd24)
        res = xuc3
        xc3 = cat([xuc3, xe12], dim=1)
        xd31 = relu(self.dec31(xc3))
        xd32 = relu(self.dec32(xd31))
        xd32 = relu(xd32+res)
        res = xd32
        xd33 = relu(self.dec33(xd32))
        xd34 = relu(self.dec34(xd33))
        xd34 = relu(xd34+res)

        # Output
        outconv = self.outconv(xd34)
        out = self.sigmoid(outconv)
        # print(f"OUTPUT: out:{out.shape}, out.dtype:{out.dtype}")

        return out


def train(
        inputs_root_dir: str = "ModelNet40_voxel_input",
        outputs_root_dir: str = "ModelNet40_ones") -> UNet:
    """Instantiates a new CNN and trains it on the given data.

    The training dataset is formed by concatenating files on the `train_files`
    list.
    """
    train_dataset = ShapeReconstructionDataset(inputs_root_dir, outputs_root_dir, skip_first_n=3, n_classes=1, every_n=1, mode="train")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=_config["batch_size"],
                                               shuffle=False, num_workers=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(train_dataset.inputs.shape)
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
        if epoch % 20 == 0 and epoch != 0:
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
            if loss.isnan():
                print(f"Loss is NaN at epoch {epoch+1}, batch {i+1}")
                print(f"Output min: {torch.min(output)}, Output mean: {torch.mean(output)} Output max: {torch.max(output)}")
                print(f"Targets min: {torch.min(targets)}, Targets mean: {torch.mean(targets)} Targets max: {torch.max(targets)}")
                break
            loss.backward()
            epoch_loss += loss
            optimizer.step()
        if not loss.isnan():
            torch.save(cnn.state_dict(), "saved_model.pt5")
        print(f"Finished epoch {epoch+1} with loss: {epoch_loss}")
    print("Finished training CnnBasic")
    return cnn

def test(
        cnn: UNet,
        inputs_root_dir: str = "ModelNet40_voxel_input",
        outputs_root_dir: str = "ModelNet40_ones") -> UNet:
    test_dataset = ShapeReconstructionDataset(inputs_root_dir, outputs_root_dir, skip_first_n=3, n_classes=1, every_n=1, mode="test")
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=_config["batch_size"],
                                               shuffle=True, num_workers=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    l1loss = nn.L1Loss()
    bceloss = nn.BCELoss()
    customloss = CustomLoss()

    cnn.eval()
    test_loss_l1 = 0
    test_loss_bce = 0
    test_loss_custom = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, targets = data
            inputs = inputs.to(device)
            targets = targets.to(device)
            output = cnn(inputs)
            loss = l1loss(output, targets)
            test_loss_l1 += loss
            loss = bceloss(output, targets)
            test_loss_bce += loss
            loss = customloss(output, targets)
            test_loss_custom += loss
        test_loss_l1 /= len(test_dataset)
        test_loss_bce /= len(test_dataset)
        test_loss_custom /= len(test_dataset)
    print(f"=== Finished testing ===")
    print(f"L1 Loss: {test_loss_l1}")
    print(f"BCE Loss: {test_loss_bce}")
    print(f"Custom Loss: {test_loss_custom}")



if __name__ == "__main__":
    cnn = train()
    torch.save(cnn.state_dict(), "saved_model.pt5")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # cnn = UNet(device, 30)
    # cnn.load_state_dict(torch.load("modelthatworkswell.pt5"))
    # cnn.load_state_dict(torch.load("saved_model.pt5"))
    test(cnn)