import argparse
import logging
import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import CNN, DeepFakeDataset  # Adjust the import as necessary


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


class Generator(nn.Module):
    def __init__(self, img_channels):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(img_channels, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, img_channels, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        perturbation = self.model(x)
        adv_x = x + perturbation
        adv_x = torch.clamp(adv_x, 0, 1)
        return adv_x


def train(
    generator,
    classifier,
    data_loader,
    optimizer,
    adversarial_loss,
    perceptual_loss,
    num_epochs,
    lambda_perceptual,
    device,
):
    generator.train()
    classifier.eval()
    for epoch in range(num_epochs):
        loop = tqdm(data_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]", leave=False)
        for imgs, labels in loop:
            imgs = imgs.to(device)
            labels = labels.to(device)

            # Generate adversarial images
            adv_imgs = generator(imgs)

            # Get classifier outputs
            outputs = classifier(adv_imgs)

            # Compute losses
            loss_adv = -adversarial_loss(outputs, labels)
            loss_perceptual = perceptual_loss(adv_imgs, imgs)
            total_loss = loss_adv + lambda_perceptual * loss_perceptual

            # Backpropagation
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Compute victim accuracy
            victim_acc = (outputs.argmax(1) == labels).float().mean()

            # Update progress bar
            loop.set_postfix(
                {
                    "Loss": total_loss.item(),
                    "Adv Loss": loss_adv.item(),
                    "Perceptual Loss": loss_perceptual.item(),
                    "Victim Acc": victim_acc.item(),
                }
            )

        logging.info(
            f"Epoch [{epoch+1}/{num_epochs}] - "
            f"Total Loss: {total_loss.item():.4f}, "
            f"Adv Loss: {loss_adv.item():.4f}, "
            f"Perceptual Loss: {loss_perceptual.item():.4f}, "
            f"Victim Accuracy: {victim_acc.item():.4f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Train an adversarial generator.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../data/140k-real-and-fake-faces/",
        help="Path to the dataset directory.",
    )
    parser.add_argument(
        "--classifier_path",
        type=str,
        required=True,
        help="Path to the pre-trained classifier checkpoint.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="adversarial_generator.pth",
        help="Path to save the generator model.",
    )
    parser.add_argument("--img_size", type=int, default=128, help="Image size.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate.")
    parser.add_argument(
        "--num_epochs", type=int, default=100, help="Number of training epochs."
    )
    parser.add_argument(
        "--lambda_perceptual",
        type=float,
        default=0.1,
        help="Weight for the perceptual loss.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    # Set random seed
    set_seed(args.seed)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Transformations
    transform = transforms.Compose(
        [transforms.Resize((args.img_size, args.img_size)), transforms.ToTensor()]
    )

    # Dataset and DataLoader
    csv_file = os.path.join(args.data_dir, "train.csv")
    root_dir = os.path.join(args.data_dir, "train")
    dataset = DeepFakeDataset(csv_file=csv_file, root_dir=root_dir, transform=transform)
    data_loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )

    # Initialize models
    classifier = CNN().to(device)
    classifier.load_state_dict(torch.load(args.classifier_path))
    classifier.eval()

    generator = Generator(img_channels=3).to(device)

    # Loss functions
    adversarial_loss = nn.CrossEntropyLoss()
    perceptual_loss = nn.MSELoss()

    # Optimizer
    optimizer_g = optim.Adam(generator.parameters(), lr=args.lr)

    # Train the generator
    train(
        generator,
        classifier,
        data_loader,
        optimizer_g,
        adversarial_loss,
        perceptual_loss,
        args.num_epochs,
        args.lambda_perceptual,
        device,
    )

    # Save the generator model
    torch.save(generator.state_dict(), args.save_path)
    logging.info(f"Generator model saved to {args.save_path}")


if __name__ == "__main__":
    main()
