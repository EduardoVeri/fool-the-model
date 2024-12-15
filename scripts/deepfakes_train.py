import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import random
import numpy as np
from torchsummary import summary
import pandas as pd
from os import path
from skimage import io, transform

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_epochs = 75
batch_size = 32
learning_rate = 0.001
path_to_data = "../data/140k-real-and-fake-faces/"


class DeepFake(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = path.join(self.root_dir, self.data.iloc[idx].loc["path"])
        image = io.imread(image_path)
        label = self.data.iloc[idx].loc["label"]

        if self.transform:
            image = self.transform(image)

        return image, label

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])


train_loader = DataLoader(
    DeepFake(
        csv_file=path.join(path_to_data, "train.csv"),
        root_dir=path_to_data,
        transform=transform
    ),
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
)

valid_loader = DataLoader(
    DeepFake(
        csv_file=path.join(path_to_data, "valid.csv"),
        root_dir=path_to_data,
        transform=transform
    ),
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
)

test_loader = DataLoader(
    DeepFake(
        csv_file=path.join(path_to_data, "test.csv"),
        root_dir=path_to_data,
        transform=transform
    ),
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
)

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


def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


model = CNN().to(device)
model.apply(initialize_weights)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


def evaluate(loader, model):
    """
    Evaluate the model on a given dataset loader.
    Returns: accuracy (%), average loss
    """
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100.0 * correct / total
    avg_loss = total_loss / len(loader)
    return accuracy, avg_loss


def train(num_epochs, model, train_loader, test_loader, optimizer, criterion):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        last_test_acc = 0.0
        for images, labels in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False
        ):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100.0 * correct / total
        train_loss = total_loss / len(train_loader)

        test_acc, test_loss = evaluate(test_loader, model)

        # after every step, save the model if the test accuracy has increased
        if test_acc > last_test_acc:
            torch.save(model.state_dict(), "simple_cnn.pth")
            last_test_acc = test_acc

        print(
            f"Epoch [{epoch+1}/{num_epochs}] - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
            f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%"
        )


print("Model Summary:")
summary(model, (3, 128, 128))

if __name__ == "__main__":
    print("Running on device:", device)
    train(num_epochs, model, train_loader, test_loader, optimizer, criterion)

    test_acc, _ = evaluate(test_loader, model)
    print(f"Final Test Accuracy: {test_acc:.2f}%")

    torch.save(model.state_dict(), "simple_cnn.pth")
    print("Model saved as simple_cnn.pth")
