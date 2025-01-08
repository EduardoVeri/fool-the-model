import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse

import torch
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataloader.dataloader import get_dataloaders
from utils import visualize_adversarial_examples, save_adversarial_examples

from arch.deepfake_cnn import CNN
from arch.adversarial_generator import MidTermGenerator


def get_args():
    parser = argparse.ArgumentParser(description="Train an adversarial generator.")
    parser.add_argument(
        "--data-dir",
        "-d",
        type=str,
        default="../../data",
        help="Path to the dataset directory.",
    )
    parser.add_argument(
        "--generator-path",
        "-g",
        type=str,
        required=True,
        help="Path to the generator checkpoint.",
    )
    parser.add_argument(
        "--output-folder",
        "-o",
        type=str,
        default=None,
    )
    parser.add_argument("--img-size", type=int, default=128, help="Image size.")
    parser.add_argument(
        "--dataset-fraction", "-df", type=float, default=0.01, help="Dataset fraction."
    )
    parser.add_argument(
        "--epsilon", "-e", type=float, default=0.1, help="Perturbation magnitude."
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

    generator = MidTermGenerator(img_channels=3, epsilon=args.epsilon).to(device)

    generator.load_state_dict(torch.load(args.generator_path))

    visualize_adversarial_examples(device, test_data_loader, generator)

    if args.output_folder is not None:
        save_adversarial_examples(
            device, test_data_loader, generator, args.output_folder
        )
        print(f"Adversarial examples saved to {args.output_folder}")


if __name__ == "__main__":
    main()
