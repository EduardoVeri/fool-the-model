import torch.nn as nn
from torch.utils.data import Dataset
import os
from skimage import io, transform
import pandas as pd

class CNN(nn.Module):
    def __init__(self, num_classes=2):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            # First Layer
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            nn.Dropout(0.05),
            # Second Layer
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            nn.Dropout(0.05),
            # Third Layer
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            nn.Dropout(0.05),
            # Fourth Layer
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            nn.Dropout(0.05),
        )
        self.classifier = nn.Sequential(
            # Flatten the kernels output
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            # First MLP Layer
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.05),
            # Second MLP Layer
            nn.Linear(128, 32),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.05),
            # Output Layer
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class DeepFakeDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.data.iloc[idx]["path"])
        image = io.imread(img_path)
        label = self.data.iloc[idx]["label"]

        if self.transform:
            image = self.transform(image)

        return image, label

