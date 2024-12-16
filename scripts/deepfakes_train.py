import argparse
import logging
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from skimage import io
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser(description="Train a CNN for DeepFake detection.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../data/140k-real-and-fake-faces/",
        help="Path to dataset directory",
    )
    parser.add_argument(
        "--csv_dir",
        type=str,
        default="../data/140k-real-and-fake-faces/",
        help="Path to CSV files directory",
    )
    parser.add_argument(
        "--save_model",
        type=str,
        default="best_cnn.pth",
        help="Path to save the best model",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=75, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument(
        "--patience", type=int, default=5, help="Early stopping patience"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def initialize_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def evaluate(loader, model, device, criterion):
    model.eval()
    correct = 0
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating", leave=False):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)
            total_samples += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100.0 * correct / total_samples
    avg_loss = total_loss / total_samples
    return accuracy, avg_loss


def train(
    model,
    train_loader,
    valid_loader,
    optimizer,
    criterion,
    device,
    num_epochs,
    patience,
    save_path,
):
    best_val_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total_samples = 0

        for images, labels in tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False
        ):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total_samples += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100.0 * correct / total_samples
        train_loss = total_loss / total_samples

        val_acc, val_loss = evaluate(valid_loader, model, device, criterion)

        logging.info(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            logging.info(f"Validation accuracy improved, model saved at {save_path}")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            logging.info(f"Early stopping after {patience} epochs with no improvement.")
            break


def main():
    args = get_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s"
    )

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ]
    )

    train_dataset = DeepFakeDataset(
        csv_file=os.path.join(args.csv_dir, "train.csv"),
        root_dir=args.data_dir,
        transform=transform,
    )
    valid_dataset = DeepFakeDataset(
        csv_file=os.path.join(args.csv_dir, "valid.csv"),
        root_dir=args.data_dir,
        transform=transform,
    )
    test_dataset = DeepFakeDataset(
        csv_file=os.path.join(args.csv_dir, "test.csv"),
        root_dir=args.data_dir,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    model = CNN().to(device)
    model.apply(initialize_weights)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    train(
        model,
        train_loader,
        valid_loader,
        optimizer,
        criterion,
        device,
        args.num_epochs,
        args.patience,
        args.save_model,
    )

    model.load_state_dict(torch.load(args.save_model))
    test_acc, _ = evaluate(test_loader, model, device, criterion)
    logging.info(f"Test Accuracy: {test_acc:.2f}%")


if __name__ == "__main__":
    main()
