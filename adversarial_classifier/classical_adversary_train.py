import argparse
import logging
import random
import os
import yaml
import joblib

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import lpips
import numpy as np

from dataloader.dataloader import get_dataloaders, DeepFakeDataset
from arch.adversarial_generator import MidTermGenerator

# Importa o classificador do scikit-learn – por exemplo, o Decision Tree
from arch.decision_tree import DTClassifier

# Como os modelos do scikit-learn não são diferenciáveis, criamos uma função wrapper
# para extrair as predições (probabilidades) e convertê-las em tensor (sem gradientes)
def classifier_forward(classifier, images):
    # images: tensor com formato [B, C, H, W] com valores no intervalo [-1, 1]
    # Normalizamos para [0,1]
    images_norm = (images + 1) / 2.0
    # Converter para numpy e reorganizar se necessário (a depender de como o dataset foi extraído)
    images_np = images_norm.detach().cpu().numpy()
    # O classificador espera cada imagem como vetor (flatten)
    B = images_np.shape[0]
    images_np = images_np.reshape(B, -1)
    # Obtemos as probabilidades usando o método predict_proba do scikit-learn
    outputs_np = classifier.predict_proba(images_np)
    # Convertemos para tensor; esses outputs estarão sem gradiente
    outputs = torch.tensor(outputs_np, device=images.device, dtype=torch.float)
    return outputs

def get_args():
    parser = argparse.ArgumentParser(
        description="Train an adversarial generator against a scikit-learn classifier."
    )
    parser.add_argument(
        "--config-path",
        "-c",
        type=str,
        default="./configs/adv_config.yaml",
        help="Path to the configuration file.",
    )
    parser.add_argument(
        "--exp-name",
        "-e",
        type=str,
        required=True,
        help="Name of the experiment to use inside the configuration file.",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the generator. If not set, the generator will be loaded from the save path and run on the test set.",
    )
    return parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def evaluate(generator, classifier, data_loader, device, verbose=False):
    generator.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in tqdm(data_loader, desc="Evaluating", disable=not verbose):
            imgs = imgs.to(device)
            labels = labels.to(device)

            adv_imgs = generator(imgs)
            # Normaliza para [0,1]
            adv_imgs_norm = (adv_imgs + 1) / 2.0
            # Utiliza o wrapper para obter as predições do classificador scikit-learn
            outputs = classifier_forward(classifier, adv_imgs)
            # Obtém as predições (a classe com maior probabilidade)
            predicted = outputs.argmax(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

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
    output_path="adversarial_generator_sl.pth",
):
    # Se houver um modelo salvo, carregamos
    if os.path.exists(output_path):
        logging.info(f"Loading existing model from {output_path}")
        generator.load_state_dict(torch.load(output_path))

    generator.train()
    # O classificador está fixo; não há gradientes propagados dele

    best_val_acc = float("inf")
    epochs_since_best = 0

    for epoch in range(num_epochs):
        # Early stopping se não houver melhora após certo número de epochs
        if epochs_since_best >= 25:
            logging.info("Early stopping after 25 epochs with no improvement.")
            break

        epoch_adv_losses = []
        epoch_perceptual_losses = []
        epoch_total_losses = []
        epoch_victim_acc = []

        loop = tqdm(train_data_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]", leave=False)
        for imgs, labels in loop:
            imgs = imgs.to(device)
            labels = labels.to(device)

            # Gera imagens adversariais
            adv_imgs = generator(imgs)
            # Normaliza para [0,1] para o classificador
            adv_imgs_norm = (adv_imgs + 1) / 2.0

            # Usa o wrapper para obter as predições do classificador (não diferenciável)
            outputs = classifier_forward(classifier, adv_imgs)
            # Cria rótulos "inversos" – por exemplo, tenta forçar a predição para a classe 1
            adv_labels = torch.ones_like(labels)

            # Calcula a adversarial loss; atenção: esta perda não propaga gradientes,
            # pois outputs vem de um classificador scikit-learn.
            loss_adv = adversarial_loss(outputs, adv_labels.long())

            # Perda perceptual (que opera diretamente nos tensores) é diferenciável
            loss_perceptual_val = torch.tensor(0.0, device=device)
            if l_perceptual > 0:
                loss_perceptual_val = l_perceptual * perceptual_loss(adv_imgs, imgs).mean()

            # Se desejar incluir pixel loss, pode somá-la (geralmente MSE entre adv_imgs e imgs)
            loss_pixel = torch.tensor(0.0, device=device)
            if l_pixel > 0:
                loss_pixel = l_pixel * pixel_loss(adv_imgs, imgs)

            total_loss = loss_adv + loss_perceptual_val + loss_pixel

            epoch_adv_losses.append(loss_adv.item())
            epoch_perceptual_losses.append(loss_perceptual_val.item())
            epoch_total_losses.append(total_loss.item())

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Como o classificador é não diferenciável, calculamos a “accuracy” do classificador separadamente
            predicted = outputs.argmax(1)
            victim_acc = (predicted == labels).float().mean()
            epoch_victim_acc.append(victim_acc.item())

            loop.set_postfix({
                "Total Loss": total_loss.item(),
                "Adv Loss": loss_adv.item(),
                "Percep Loss": loss_perceptual_val.item(),
                "Victim Acc": victim_acc.item() * 100,
            })

        # Avalia no conjunto de validação
        val_acc = evaluate(generator, classifier, val_data_loader, device, verbose=True)
        logging.info(f"Epoch [{epoch+1}/{num_epochs}] - Validation Accuracy: {val_acc:.2f}%")
        epochs_since_best += 1
        if val_acc < best_val_acc:
            best_val_acc = val_acc
            epochs_since_best = 0
            torch.save(generator.state_dict(), output_path)
            logging.info(f"New best model saved with validation accuracy: {val_acc:.2f}%")

        mean_adv_loss = np.mean(epoch_adv_losses)
        mean_perc_loss = np.mean(epoch_perceptual_losses)
        mean_total_loss = np.mean(epoch_total_losses)
        mean_victim_acc = np.mean(epoch_victim_acc) * 100

        logging.info(
            f"Epoch [{epoch+1}/{num_epochs}] - Total Loss: {mean_total_loss:.4f}, "
            f"Adv Loss: {mean_adv_loss:.4f}, Perceptual Loss: {mean_perc_loss:.4f}, "
            f"Victim Acc: {mean_victim_acc:.2f}%"
        )

def main():
    args = get_args()

    # Carrega configurações do arquivo YAML
    with open(args.config_path, "r") as file:
        config = yaml.safe_load(file)
    exp_config = config[args.exp_name]

    img_size = exp_config["img-size"]
    batch_size = exp_config["batch-size"]
    dataset_fraction = exp_config["dataset-fraction"]
    num_epochs = exp_config["epochs"]
    lr = exp_config["lr"]
    lambda_perceptual = exp_config["lper"]
    lambda_pixel = exp_config.get("lpixel", 0)
    epsilon = exp_config["epsilon"]
    seed = exp_config["seed"]
    data_dir = exp_config["data-dir"]
    classifier_path = exp_config["classifier-path"]
    save_path = exp_config["save-path"]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("adversarial_generator_sl.log"),
            logging.StreamHandler(),
        ],
    )

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Transformação similar à utilizada no treinamento da adversarial original
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # Carrega os dataloaders
    train_data_loader, val_data_loader, test_data_loader = get_dataloaders(
        data_dir, batch_size, transform, dataset_fraction
    )

    # Carrega o classificador scikit-learn usando joblib
    classifier = joblib.load(classifier_path)
    # Observe que não há necessidade de mover o classificador para o device

    # Inicializa o gerador adversarial (modelo GAN fixo)
    generator = MidTermGenerator(img_channels=3, epsilon=epsilon).to(device)

    if args.train:
        adversarial_loss = nn.CrossEntropyLoss()
        perceptual_loss = lpips.LPIPS(net="vgg").to(device)
        pixel_loss_fn = nn.MSELoss()
        optimizer_g = optim.Adam(generator.parameters(), lr=lr, weight_decay=1e-5)

        logging.info("Start training the adversarial generator (scikit-learn classifier)...")
        try:
            train(
                generator=generator,
                classifier=classifier,
                train_data_loader=train_data_loader,
                val_data_loader=val_data_loader,
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
            torch.save(generator.state_dict(), "adversarial_generator_sl_interrupted.pth")
    else:
        logging.info("Loading existing adversarial generator...")
        generator.load_state_dict(torch.load(save_path))

    # Exemplo de visualização de uma imagem adversarial
    fake_image, _ = DeepFakeDataset(
        csv_file=os.path.join(data_dir, "test.csv"),
        root_dir=data_dir,
        transform=transform,
    ).get_real_image(745)
    # Visualize uma imagem adversarial
    from utils import visualize_one_adversarial_example
    visualize_one_adversarial_example(device, fake_image, generator)

if __name__ == "__main__":
    main()
