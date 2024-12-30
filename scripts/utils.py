import torch.nn as nn
from torch.utils.data import Dataset
import os
from skimage import io, transform
import pandas as pd

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
    def __init__(self, csv_file, root_dir, fraction = 1, transform=None):
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

