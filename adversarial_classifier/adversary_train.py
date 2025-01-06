import argparse
import logging
import random
from os import path
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import lpips

from dataloader.dataloader import get_dataloaders

from arch.deepfake_cnn import CNN
from arch.adversarial_generator import MidTermGenerator

from utils import visualize_adversarial_examples


def get_args():
    parser = argparse.ArgumentParser(description="Train an adversarial generator.")
    parser.add_argument(
        "--data-dir",
        "-d",
        type=str,
        default="../data",
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
        "-s",
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
        "-lper",
        type=float,
        default=0.1,
        help="Weight for the perceptual loss.",
    )
    parser.add_argument(
        "--lambda-pixel",
        "-lp",
        type=float,
        default=0.1,
        help="Weight for the pixel loss.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--dataset-fraction", type=float, default=1.0, help="Dataset fraction."
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the generator. If not set, the generator will be loaded from the save path.",
    )
    parser.add_argument(
        "--epsilon",
        "-e",
        type=float,
        default=0.1,
        help="Maximum perturbation allowed in the L infinity norm.",
    )
    args = parser.parse_args()

    return args


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def evaluate(generator, classifier, data_loader, device, confusion_matrix=False):
    generator.eval()
    classifier.eval()
    correct = 0
    total = 0

    false_positives = 0
    false_negatives = 0
    true_positives = 0
    true_negatives = 0

    with torch.no_grad():
        for imgs, labels in data_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            adv_imgs = generator(imgs)
            adv_imgs_norm = (adv_imgs + 1) / 2  # Normalize to [0, 1]
            outputs = classifier(adv_imgs_norm)

            total += labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()

            if confusion_matrix:
                outputs = outputs.argmax(1)
                for output, label in zip(outputs, labels):
                    if output == 1 and label == 1:
                        true_positives += 1
                    elif output == 1 and label == 0:
                        false_positives += 1
                    elif output == 0 and label == 1:
                        false_negatives += 1
                    else:
                        true_negatives += 1

    if confusion_matrix:
        # print confusion matrix
        cm = (
            "\nConfusion Matrix:\n"
            f"{'':<20} | {'Predicted 0':<15} | {'Predicted 1':<15}\n"
            f"{'-'*20}-+-{'-'*15}-+-{'-'*15}\n"
            f"{'Actual 0':<20} | {true_negatives:<15} | {false_positives:<15}\n"
            f"{'Actual 1':<20} | {false_negatives:<15} | {true_positives:<15}"
        )
        logging.info(cm)

    return 100 * correct / total


def train(
    generator,
    classifier,
    train_data_loader,
    val_data_loader,
    optimizer,
    adversarial_loss,
    perceptual_loss,
    pixel_loss,
    num_epochs,
    l_perceptual,
    l_pixel,
    device,
    output_path=None,
    epsilon=0.1,
):

    generator.train()
    classifier.eval()
    loss_adv_list = []
    loss_perceptual_list = []
    loss_pixel_list = []
    total_loss_list = []
    best_val_acc = float("+inf")
    last_best_epoch = 0

    for epoch in range(num_epochs):
        # Early stopping
        if last_best_epoch >= 25:
            logging.info("Early stopping after 5 epochs with no improvement.")
            break

        epoch_adv_list = []
        epoch_perceptual_list = []
        epoch_total_list = []
        epoch_victim_acc_list = []

        loop = tqdm(
            train_data_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]", leave=False
        )
        for imgs, labels in loop:
            imgs = imgs.to(device)
            labels = labels.to(device)

            # Constrain the perturbation in L infinity norm
            adv_imgs = generator.clamp_perturbation(imgs, generator(imgs), epsilon)

            # Get classifier outputs
            adv_imgs_norm = (adv_imgs + 1) / 2.0  # map [-1,1] -> [0,1]
            outputs = classifier(adv_imgs_norm)

            # Invert the labels for the adversarial loss
            adv_labels = 1 - labels  # binary classification

            # Compute losses
            loss_adv = adversarial_loss(outputs, adv_labels.long())
            loss_perceptual = l_perceptual * perceptual_loss(adv_imgs, imgs).mean()
            # loss_px = l_pixel * pixel_loss(adv_imgs, imgs)

            total_loss = loss_adv + loss_perceptual  # + loss_px

            # Logging values
            epoch_adv_list.append(loss_adv.item())
            epoch_perceptual_list.append(loss_perceptual.item())
            # loss_pixel_list.append(loss_px.item())
            epoch_total_list.append(total_loss.item())

            # Backprop + Step
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Compute victim accuracy
            victim_acc = (outputs.argmax(1) == labels).float().mean()
            epoch_victim_acc_list.append(victim_acc.item())

            # Update progress bar
            loop.set_postfix(
                {
                    "Loss": total_loss.item(),
                    "Adv Loss": loss_adv.item(),
                    "Perceptual Loss": loss_perceptual.item(),
                    "Victim Acc": victim_acc.item(),
                }
            )

        # Evaluate on validation
        val_acc = evaluate(generator, classifier, val_data_loader, device)
        logging.info(f"Validation Accuracy: {val_acc:.2f}%")

        # Save model if improved
        last_best_epoch += 1
        if val_acc < best_val_acc:
            best_val_acc = val_acc
            last_best_epoch = 0
            torch.save(
                generator.state_dict(),
                path.join(output_path, f"E{epsilon:-1.2f}_epoch{epoch}_generator.pth"),
            )
            logging.info(
                f"New best model saved with validation accuracy: {val_acc:.2f}%"
            )

        mean_loss_adv = sum(epoch_adv_list) / len(epoch_adv_list)
        mean_perceptual_loss = sum(epoch_perceptual_list) / len(epoch_perceptual_list)
        mean_loss_total = sum(epoch_total_list) / len(epoch_total_list)
        mean_victim_acc = (
            sum(epoch_victim_acc_list) / len(epoch_victim_acc_list)
        ) * 100

        loss_adv_list.append(mean_loss_adv)
        loss_perceptual_list.append(mean_perceptual_loss)
        total_loss_list.append(mean_loss_total)

        logging.info(
            f"Epoch [{epoch+1}/{num_epochs}] - "
            f"Total Loss: {mean_loss_total:.4f}, "
            f"Adv Loss: {mean_loss_adv:.4f}, "
            f"Perceptual Loss: {mean_perceptual_loss:.4f}, "
            f"Victim Acc: {mean_victim_acc:.2f}, "
            f"Validation Acc: {val_acc:.2f}%"
        )

    # Plot losses
    plt.plot(loss_adv_list, label="Adversarial Loss")
    plt.plot(loss_perceptual_list, label="Perceptual Loss")
    plt.plot(total_loss_list, label="Total Loss")
    plt.legend()
    plt.show()


def main():
    args = get_args()

    if not path.exists(args.save_path):
        os.makedirs(args.save_path)

    epsilon = args.epsilon
    
    # Configure logging in a file
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("adversarial_generator.log"),
            logging.StreamHandler(),
        ],
    )

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
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    # Dataset and DataLoader
    train_data_loader, valid_data_loader, test_data_loader = get_dataloaders(
        args.data_dir, args.batch_size, transform, args.dataset_fraction
    )

    # Initialize models
    classifier = CNN().to(device)
    classifier.load_state_dict(torch.load(args.classifier_path, weights_only=True))
    classifier.eval()

    generator = MidTermGenerator(img_channels=3).to(device)

    if args.train:
        # Loss functions
        adversarial_loss = nn.CrossEntropyLoss()
        perceptual_loss = lpips.LPIPS(net="vgg").to(device)
        pixel_loss_fn = nn.MSELoss()

        # Optimizer
        optimizer_g = optim.Adam(generator.parameters(), lr=args.lr, weight_decay=1e-5)

        logging.info("Start training the adversarial generator...")

        # Train the generator
        train(
            generator=generator,
            classifier=classifier,
            train_data_loader=train_data_loader,
            val_data_loader=valid_data_loader,
            optimizer=optimizer_g,
            adversarial_loss=adversarial_loss,
            perceptual_loss=perceptual_loss,
            pixel_loss=pixel_loss_fn,
            num_epochs=args.num_epochs,
            l_perceptual=args.lambda_perceptual,
            l_pixel=args.lambda_pixel,
            device=device,
            output_path=args.save_path,
            epsilon=epsilon,
        )
    else:
        generator.load_state_dict(torch.load(args.save_path))

    # Check Accuracy in the test set after training
    test_acc = evaluate(generator, classifier, test_data_loader, device, True)
    logging.info(f"Test Accuracy: {test_acc:.2f}%")

    # Show some images generated by the generator
    generator.eval()

    visualize_adversarial_examples(device, test_data_loader, generator, epsilon)


if __name__ == "__main__":
    main()
