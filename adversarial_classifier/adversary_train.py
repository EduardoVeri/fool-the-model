import argparse
import logging
import random
from os import path
import os
import yaml

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
        "--config-path",
        "-c",
        type=str,
        required=True,
        help="Path to the configuration file.",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the generator. If not set, the generator will be loaded from the save path.",
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
    output_path="adversarial_generator.pth",
):

    if os.path.exists(output_path):
        logging.info(f"Loading existing model in output path {output_path}")
        generator.load_state_dict(torch.load(output_path))

    generator.train()
    classifier.eval()

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
            adv_imgs = generator(imgs)

            # Get classifier outputs
            adv_imgs_norm = (adv_imgs + 1) / 2.0  # map [-1,1] -> [0,1]
            outputs = classifier(adv_imgs_norm)

            # Invert the labels for the adversarial loss
            adv_labels = torch.ones_like(labels)

            # Compute losses
            loss_adv = adversarial_loss(outputs, adv_labels.long())
            loss_perceptual_val = torch.tensor(0.0, device=device)
            if l_perceptual > 0:
                loss_perceptual_val = (
                    l_perceptual * perceptual_loss(adv_imgs, imgs).mean()
                )
            total_loss = loss_adv + loss_perceptual_val

            # Logging values
            epoch_adv_list.append(loss_adv.item())
            if l_perceptual > 0:
                epoch_perceptual_list.append(loss_perceptual_val.item())
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
                    "Perceptual Loss": loss_perceptual_val.item(),
                    "Victim Acc": victim_acc.item() * 100,
                }
            )

        # Evaluate on validation
        val_acc = evaluate(generator, classifier, val_data_loader, device, True)
        logging.info(f"Validation Accuracy: {val_acc:.2f}%")

        # Save model if improved
        last_best_epoch += 1
        if val_acc < best_val_acc:
            best_val_acc = val_acc
            last_best_epoch = 0
            torch.save(generator.state_dict(), output_path)
            logging.info(
                f"New best model saved with validation accuracy: {val_acc:.2f}%"
            )

        mean_loss_adv = sum(epoch_adv_list) / len(epoch_adv_list)
        mean_perceptual_loss = (
            0
            if len(epoch_perceptual_list) == 0
            else sum(epoch_perceptual_list) / len(epoch_perceptual_list)
        )
        mean_loss_total = sum(epoch_total_list) / len(epoch_total_list)
        mean_victim_acc = (
            sum(epoch_victim_acc_list) / len(epoch_victim_acc_list)
        ) * 100

        logging.info(
            f"Epoch [{epoch+1}/{num_epochs}] - "
            f"Total Loss: {mean_loss_total:.4f}, "
            f"Adv Loss: {mean_loss_adv:.4f}, "
            f"Perceptual Loss: {mean_perceptual_loss:.4f}, "
            f"Victim Acc: {mean_victim_acc:.2f}%, "
            f"Validation Acc: {val_acc:.2f}%"
        )


def main():
    args = get_args()

    # Get configs from yaml file
    with open(args.config_path, "r") as file:
        config = yaml.safe_load(file)
    
    img_size = config["img_size"]
    batch_size = config["batch_size"]
    dataset_fraction = config["dataset_fraction"]
    num_epochs = config["num_epochs"]
    lr = config["lr"]
    lambda_perceptual = config["lambda_perceptual"]
    lambda_pixel = config["lambda_pixel"]
    epsilon = config["epsilon"]
    seed = config["seed"]
    data_dir = config["data_dir"]
    classifier_path = config["classifier_path"]
    save_path = config["save_path"]
    
    
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
    set_seed(seed)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Transformations
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    # Dataset and DataLoader
    train_data_loader, valid_data_loader, test_data_loader = get_dataloaders(
        data_dir, batch_size, transform, dataset_fraction
    )

    # Initialize models
    classifier = CNN().to(device)
    classifier.load_state_dict(torch.load(classifier_path, weights_only=True))
    classifier.eval()

    for param in classifier.parameters():
        param.requires_grad = False

    generator = MidTermGenerator(img_channels=3, epsilon=epsilon).to(device)

    if args.train:
        # Loss functions
        adversarial_loss = nn.CrossEntropyLoss()
        perceptual_loss = lpips.LPIPS(net="vgg").to(device)
        pixel_loss_fn = nn.MSELoss()

        # Optimizer
        optimizer_g = optim.Adam(generator.parameters(), lr=lr, weight_decay=1e-5)

        logging.info("Start training the adversarial generator...")

        # Train the generator
        try:
            train(
                generator=generator,
                classifier=classifier,
                train_data_loader=train_data_loader,
                val_data_loader=valid_data_loader,
                optimizer=optimizer_g,
                adversarial_loss=adversarial_loss,
                perceptual_loss=perceptual_loss,
                pixel_loss=pixel_loss_fn,
                num_epochs=num_epochs,
                l_perceptual=lambda_perceptual,
                l_pixel=lambda_pixel,
                device=device,
                output_path=save_path,
            )
        except KeyboardInterrupt:
            logging.info("Training interrupted. Saving model...")
            torch.save(generator.state_dict(), "adversarial_generator_interrupted.pth")

    else:
        logging.info("Loading existing model...")
        logging.info("If you want to train the generator, use the --train flag.")
        
        generator.load_state_dict(torch.load(save_path))

    # Check Accuracy in the test set after training
    test_acc = evaluate(generator, classifier, test_data_loader, device, True)
    logging.info(f"Test Accuracy: {test_acc:.2f}%")

    # Show some images generated by the generator
    visualize_adversarial_examples(device, test_data_loader, generator)


if __name__ == "__main__":
    main()
