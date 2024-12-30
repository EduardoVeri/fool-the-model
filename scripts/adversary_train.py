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
import matplotlib.pyplot as plt
import lpips

from utils import CNN, DeepFakeDataset  # Adjust the import as necessary


def get_args():
    parser = argparse.ArgumentParser(description="Train an adversarial generator.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="../data/140k-real-and-fake-faces/",
        help="Path to the dataset directory.",
    )
    parser.add_argument(
        "--classifier-path",
        type=str,
        required=True,
        help="Path to the pre-trained classifier checkpoint.",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="adversarial_generator.pth",
        help="Path to save the generator model.",
    )
    parser.add_argument("--img-size", type=int, default=128, help="Image size.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size.")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate.")
    parser.add_argument(
        "--num-epochs", type=int, default=10, help="Number of training epochs."
    )
    parser.add_argument(
        "--lambda-perceptual",
        type=float,
        default=0.1,
        help="Weight for the perceptual loss.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    return args


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
            # nn.Dropout(0.3),
            nn.Conv2d(64, img_channels, kernel_size=3, padding=1),
            # nn.Tanh(),
        )

    def forward(self, x):
        perturbation = self.model(x)
        adv_x = x + perturbation
        adv_x = torch.clamp(adv_x, -1, 1)
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
    loss_adv_list = []
    loss_perceptual_list = []
    total_loss_list = []
    for epoch in range(num_epochs):
        loop = tqdm(data_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]", leave=False)
        for imgs, labels in loop:
            imgs = imgs.to(device)
            labels = labels.to(device)

            # Generate adversarial images
            adv_imgs = generator(imgs)

            # Get classifier outputs
            outputs = classifier(adv_imgs)

            # Invert the labels for the adversarial loss
            adv_labels = 1 - labels

            # Compute losses
            loss_adv = adversarial_loss(outputs, adv_labels)
            loss_perceptual = perceptual_loss.forward(adv_imgs, imgs).mean()
            total_loss = loss_adv + lambda_perceptual * loss_perceptual
            
            loss_adv_list.append(loss_adv.item())
            loss_perceptual_list.append(loss_perceptual.item())
            total_loss_list.append(total_loss.item())

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

    plt.plot(loss_adv_list, label="Adversarial Loss")
    plt.plot(loss_perceptual_list, label="Perceptual Loss")
    plt.plot(total_loss_list, label="Total Loss")
    plt.legend()
    plt.show()


def main():
    args = get_args()

    # Set random seed
    set_seed(args.seed)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Transformations
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]
    )

    # Dataset and DataLoader
    csv_file = os.path.join(args.data_dir, "train.csv")
    root_dir = os.path.join(args.data_dir)
    dataset = DeepFakeDataset(
        csv_file=csv_file, root_dir=root_dir, transform=transform, fraction=0.01
    )
    data_loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )

    # Initialize models
    classifier = CNN().to(device)
    classifier.load_state_dict(torch.load(args.classifier_path, weights_only=True))
    classifier.eval()

    generator = Generator(img_channels=3).to(device)

    # Loss functions
    adversarial_loss = nn.CrossEntropyLoss()
    # perceptual_loss = nn.MSELoss()
    perceptual_loss = lpips.LPIPS(net="vgg").to(device)

    # Optimizer
    optimizer_g = optim.Adam(generator.parameters(), lr=args.lr, weight_decay=1e-5)

    # Configure logging in a file
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("adversarial_generator.log"),
            logging.StreamHandler(),
        ],
    )
    logging.info("Start training the adversarial generator...")

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
    
    
    # Show some images generated by the generator
    generator.eval()
    imgs, _ = next(iter(data_loader))
    
    
    inv_transform = transforms.Compose([
        transforms.Normalize(
            mean=[-0.5/0.5, -0.5/0.5, -0.5/0.5],  # or [-1, -1, -1]
            std=[1/0.5, 1/0.5, 1/0.5]            # or [2, 2, 2]
        )
    ])
    
    imgs = imgs.to(device)
    adv_imgs = generator(imgs).detach().cpu()
    adv_imgs = inv_transform(adv_imgs)
    imgs = inv_transform(imgs.detach().cpu())
    
    fig, ax = plt.subplots(2, 5, figsize=(20, 8))
    for i in range(5):
        ax[0, i].imshow(imgs[i].permute(1, 2, 0))
        ax[0, i].axis("off")
        ax[1, i].imshow(adv_imgs[i].permute(1, 2, 0))
        ax[1, i].axis("off")
    plt.show()
    

    # Save the generator model
    torch.save(generator.state_dict(), args.save_path)
    logging.info(f"Generator model saved to {args.save_path}")


if __name__ == "__main__":
    main()
