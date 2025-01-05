import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse

import torch
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataloader.dataloader import get_dataloaders

from arch.deepfake_cnn import CNN
from arch.adversarial_generator import MidTermGenerator


def get_args():
    parser = argparse.ArgumentParser(description="Train an adversarial generator.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="../../data/140k-real-and-fake-faces/",
        help="Path to the dataset directory.",
    )
    parser.add_argument(
        "--classifier-path",
        type=str,
        required=True,
        help="Path to the pre-trained classifier checkpoint.",
    )
    parser.add_argument(
        "--generator-path",
        type=str,
        required=True,
        help="Path to the generator checkpoint.",
    )
    parser.add_argument("--img-size", type=int, default=128, help="Image size.")

    parser.add_argument(
        "--dataset-fraction", type=float, default=1.0, help="Dataset fraction."
    )
    args = parser.parse_args()

    return args




def main():
    args = get_args()

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Transformations
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    # Dataset and DataLoader
    _, _, test_data_loader = get_dataloaders(
        args.data_dir, 1, transform, args.dataset_fraction
    )

    # Initialize models
    classifier = CNN().to(device)
    classifier.load_state_dict(torch.load(args.classifier_path, weights_only=True))
    classifier.eval()

    generator = MidTermGenerator(img_channels=3).to(device)


    generator.load_state_dict(torch.load(args.generator_path))


    # Show some images generated by the generator
    generator.eval()

    inv_transform = transforms.Compose(
        [transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2])]
    )

    fig, ax = plt.subplots(2, 5, figsize=(20, 8))
    for i, (img, label) in zip(range(5), test_data_loader):
        img = img.to(device)
        adv_img = generator(img).detach().cpu()
        adv_img = inv_transform(adv_img)
        img = inv_transform(img.detach().cpu())
        ax[0, i].set_title(label[0])
        ax[0, i].imshow(img[0].permute(1, 2, 0))
        ax[0, i].axis("off")
        ax[1, i].set_title(label[0])
        ax[1, i].imshow(img[0].permute(1, 2, 0))
        ax[1, i].axis("off")
    plt.show()


if __name__ == "__main__":
    main()
