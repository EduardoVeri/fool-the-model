import argparse
import logging
import os
import random
import numpy as np
import joblib

from torchvision import transforms
from torch.utils.data import DataLoader
from dataloader.dataloader import DeepFakeDataset
from arch.decision_tree import DTClassifier as DT
from arch.gradient_tree_boosting import GTBClassifier as GTB

def get_args():
    parser = argparse.ArgumentParser(
        description="Train a Decision Tree for DeepFake detection using scikit-learn."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../data/",
        help="Path to dataset directory",
    )
    parser.add_argument(
        "--csv_dir",
        type=str,
        default="../data/",
        help="Path to CSV files directory",
    )
    parser.add_argument(
        "--save_model",
        type=str,
        default="best_dt.pkl",
        help="Path to save the best model",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for data extraction (not training)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def extract_features(dataset):
    """
    Converte as imagens do dataset em vetores unidimensionais.
    Cada imagem com formato (C, H, W) é transformada em um vetor de features.
    """
    X = []
    y = []
    for image, label in dataset:
        # image é um tensor; convertemos para numpy e realizamos o flatten
        np_img = image.numpy()
        X.append(np_img.flatten())
        y.append(label)
    return np.array(X), np.array(y)

def main():
    args = get_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s")
    set_seed(args.seed)

    # Define a transformação para as imagens, similar à utilizada em deepfakes_train.py
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    # Carrega os datasets
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

    # Extração de features (flatten das imagens)
    logging.info("Extraindo features do conjunto de treino")
    X_train, y_train = extract_features(train_dataset)
    logging.info("Extraindo features do conjunto de validação")
    X_valid, y_valid = extract_features(valid_dataset)
    logging.info("Extraindo features do conjunto de teste")
    X_test, y_test = extract_features(test_dataset)

    # Instancia o modelo Decision Tree
    dt_model = DT(random_state=args.seed)
    dt_model = GTB(random_state=args.seed)

    # Otimização dos hiperparâmetros via GridSearchCV
    param_grid = {
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    logging.info("Realizando tuning dos hiperparâmetros com GridSearchCV")
    best_params, best_score = dt_model.tune_parameters(param_grid, X_train, y_train, cv=5)
    logging.info(f"Melhores parâmetros: {best_params} com acurácia de validação: {best_score:.2f}")

    # Treina o modelo com o conjunto completo de treino
    dt_model.fit(X_train, y_train)

    # Avalia o modelo nos conjuntos de validação e teste
    valid_acc = dt_model.score(X_valid, y_valid)
    logging.info(f"Acurácia na validação: {valid_acc * 100:.2f}%")
    test_acc = dt_model.score(X_test, y_test)
    logging.info(f"Acurácia no teste: {test_acc * 100:.2f}%")

    # Salva o modelo treinado
    joblib.dump(dt_model, args.save_model)
    logging.info(f"Modelo salvo em {args.save_model}")

if __name__ == "__main__":
    main()
