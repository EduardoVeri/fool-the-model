import torch
import torch.nn as nn


class SmallUNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=2, features=32):
        super(SmallUNet, self).__init__()
        # Encoder
        self.enc1 = self.conv_block(in_channels, features)  # [B, features, 128, 128]
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # [B, features, 64, 64]
        self.enc2 = self.conv_block(features, features * 2)  # [B, features*2, 64, 64]
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # [B, features*2, 32, 32]

        # Bottleneck
        self.bottleneck = self.conv_block(
            features * 2, features * 4
        )  # [B, features*4, 32, 32]

        # Decoder
        self.up1 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )  # [B, features*2, 64, 64]
        self.dec1 = self.conv_block(features * 4, features * 2)  # combining with enc2
        self.up2 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )  # [B, features, 128, 128]
        self.dec2 = self.conv_block(features * 2, features)  # combining with enc1

        # Classification head
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(features, num_classes)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1),
        )

    def forward(self, x):
        # Encoder path
        enc1 = self.enc1(x)  # Shape: [B, features, 128, 128]
        enc2 = self.enc2(self.pool1(enc1))  # Shape: [B, features*2, 64, 64]

        # Bottleneck
        bottleneck = self.bottleneck(self.pool2(enc2))  # Shape: [B, features*4, 32, 32]

        # Decoder path with skip connections
        up1 = self.up1(bottleneck)  # Shape: [B, features*2, 64, 64]
        # Concatenate skip connection from enc2 along the channel dimension
        dec1 = self.dec1(
            torch.cat([up1, enc2], dim=1)
        )  # Shape: [B, features*2, 64, 64]
        up2 = self.up2(dec1)  # Shape: [B, features, 128, 128]
        # Concatenate skip connection from enc1
        dec2 = self.dec2(
            torch.cat([up2, enc1], dim=1)
        )  # Shape: [B, features, 128, 128]

        # Classification head
        pooled = self.global_pool(dec2)  # Shape: [B, features, 1, 1]
        pooled = pooled.view(pooled.size(0), -1)  # Flatten to [B, features]
        logits = self.fc(pooled)  # Final logits: [B, num_classes]
        return logits
