import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
from skimage import io, transform
import pandas as pd
import torch
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, num_classes=2):
        super(CNN, self).__init__()

        self.relu = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(0.05)

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.batchnorm3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.batchnorm4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.global_max_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.dropout(self.relu(self.batchnorm1(self.conv1(x))))
        x = self.pool1(x)
        x = self.dropout(self.relu(self.batchnorm2(self.conv2(x))))
        x = self.pool2(x)
        x = self.dropout(self.relu(self.batchnorm3(self.conv3(x))))
        x = self.pool3(x)
        x = self.dropout(self.relu(self.batchnorm4(self.conv4(x))))
        x = self.pool4(x)
        x = self.global_max_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.fc2(x))
        x = self.dropout(self.fc3(x))

        return x


class DeepFakeDataset(Dataset):
    def __init__(self, csv_file, root_dir, fraction=1, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.data = self.data.sample(frac=fraction).reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.data.iloc[idx]["path"])
        image = io.imread(img_path)
        label = self.data.iloc[idx]["label"]

        if self.transform:
            image = self.transform(image)

        return image, label


class ResidualBlock(nn.Module):
    """
    A simple residual block to help the network learn identity mappings.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # If in_channels != out_channels, use a 1×1 conv to match dims
        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
            if in_channels != out_channels
            else None
        )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.skip is not None:
            identity = self.skip(identity)

        out += identity
        out = self.relu(out)
        return out


class DownBlock(nn.Module):
    """
    Downsampling block: Conv -> BN -> ReLU -> (optional residual) -> 2×2 avg pool
    """

    def __init__(self, in_channels, out_channels, use_residual=False):
        super().__init__()
        self.use_residual = use_residual

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        if use_residual:
            self.res = ResidualBlock(out_channels, out_channels)
        self.pool = nn.AvgPool2d(kernel_size=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        if self.use_residual:
            x = self.res(x)
        x = self.pool(x)
        return x


class UpBlock(nn.Module):
    """
    Upsampling block: nearest-neighbor upsample -> Conv -> BN -> ReLU -> (optional residual)
    """

    def __init__(self, in_channels, out_channels, use_residual=False):
        super().__init__()
        self.use_residual = use_residual

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        if use_residual:
            self.res = ResidualBlock(out_channels, out_channels)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        if self.use_residual:
            x = self.res(x)
        return x


class MidTermGenerator(nn.Module):
    """
    A mid-size generator that transforms an input image x in [-1,1] space
    into a perturbation (also in [-1,1]), then outputs the adversarial image.
    The final output is clamped to [-1,1].

    Architecture:
      Downsample (128->64->32)
      Bottleneck residual block(s)
      Upsample (32->64->128)
    """

    def __init__(self, img_channels=3, base_channels=32):
        super().__init__()

        # -----------------------
        #     Encoder (down)
        # -----------------------
        self.down1 = DownBlock(img_channels, base_channels, use_residual=False)
        self.down2 = DownBlock(base_channels, base_channels * 2, use_residual=True)

        # Bottleneck
        self.bottleneck = ResidualBlock(base_channels * 2, base_channels * 2)

        # -----------------------
        #     Decoder (up)
        # -----------------------
        self.up1 = UpBlock(base_channels * 2, base_channels, use_residual=True)
        self.up2 = UpBlock(base_channels, base_channels, use_residual=False)

        # Final conv to predict a perturbation
        self.final_conv = nn.Conv2d(
            base_channels, img_channels, kernel_size=3, padding=1
        )

    def forward(self, x):
        """
        x: in [-1,1], shape [B, 3, 128, 128]
        returns adv_x in [-1,1], same shape
        """
        # Encoder
        d1 = self.down1(x)  # 128 -> 64
        d2 = self.down2(d1)  # 64 -> 32

        # Bottleneck
        b = self.bottleneck(d2)

        # Decoder
        u1 = self.up1(b)  # 32 -> 64
        u2 = self.up2(u1)  # 64 -> 128

        # Final perturbation
        perturbation = self.final_conv(u2)  # no activation -> can be any real
        # If you want a narrower range, you can add a .tanh() here
        # but often it's better to clamp after you add to x.

        adv_x = x + perturbation
        adv_x = torch.clamp(adv_x, -1, 1)
        return adv_x


def get_dataloaders(path: str, batch_size: int, transform: object, fraction: float = 1):
    train_dataset = DeepFakeDataset(
        csv_file=os.path.join(path, "train.csv"),
        root_dir=path,
        fraction=fraction,
        transform=transform,
    )
    valid_dataset = DeepFakeDataset(
        csv_file=os.path.join(path, "valid.csv"),
        root_dir=path,
        fraction=fraction,
        transform=transform,
    )
    test_dataset = DeepFakeDataset(
        csv_file=os.path.join(path, "test.csv"),
        root_dir=path,
        fraction=fraction,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, valid_loader, test_loader
