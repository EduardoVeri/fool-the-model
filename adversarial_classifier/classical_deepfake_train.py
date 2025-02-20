import argparse
import logging
import os
import random
import numpy as np
import joblib
import tqdm

from torchvision import transforms
from torch.utils.data import DataLoader
from dataloader.dataloader import DeepFakeDataset
from arch.decision_tree import DTClassifier as DT
from arch.gradient_tree_boosting import GTBClassifier as GTB
from arch.xgboost_classifier import XGBClassifierWrapper as XGB

import cv2
import tqdm
import pywt
from skimage.feature import hog, local_binary_pattern
from concurrent.futures import ProcessPoolExecutor, as_completed

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
    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="Fraction of the dataset to use (for debugging)",
    )
    return parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def extract_wavelet_features(gray_img, wavelet='haar', level=2):
    """
    Extracts wavelet features by performing a multilevel 2D wavelet decomposition on a grayscale image.
    It computes the mean and standard deviation of the approximation coefficients and each set of detail coefficients.
    
    Parameters:
        gray_img: 2D numpy array (grayscale image) of type uint8.
        wavelet: Type of wavelet to use.
        level: Decomposition level.
        
    Returns:
        A 1D numpy array of wavelet features.
    """
    coeffs = pywt.wavedec2(gray_img, wavelet, level=level)
    features = []
    # Approximation coefficients at the coarsest level
    cA = coeffs[0]
    features.append(np.mean(cA))
    features.append(np.std(cA))
    # Process each level's detail coefficients (horizontal, vertical, diagonal)
    for detail_level in coeffs[1:]:
        cH, cV, cD = detail_level
        features.append(np.mean(cH))
        features.append(np.std(cH))
        features.append(np.mean(cV))
        features.append(np.std(cV))
        features.append(np.mean(cD))
        features.append(np.std(cD))
    return np.array(features)

def process_indices(dataset, indices, 
                    hog_orientations=9, 
                    hog_pixels_per_cell=(8, 8), 
                    hog_cells_per_block=(2, 2),
                    lbp_P=8, 
                    lbp_R=1, 
                    lbp_method='uniform',
                    wavelet='haar',
                    wavelet_level=2):
    """
    Processes a list of indices from the dataset.
    For each index, it retrieves the image and label using dataset[index]
    and extracts a combined feature vector consisting of:
      - HOG features (edge orientation)
      - LBP features (local texture)
      - Wavelet features (multi-scale frequency information)
      - Color Histogram features (color distribution)
      - Hu Moments (global shape descriptors)
      
    The function is designed to work with images that are either channels-first (3, H, W) 
    or channels-last (H, W, 3). It maintains parallel execution by processing individual indices.
    """
    X_chunk = []
    y_chunk = []
    
    # For LBP with 'uniform' method, the number of bins is P + 2.
    lbp_n_bins = lbp_P + 2

    # Use tqdm progress bar only for the first chunk.
    if indices[0] == 0:
        loop = tqdm.tqdm(indices, desc="Processing indices")
    else:
        loop = indices
    
    for idx in loop:
        # Retrieve image and label using __getitem__
        image, label = dataset[idx]
        
        # If the image is a tensor, convert to numpy array.
        if hasattr(image, 'numpy'):
            np_img = image.numpy()
        else:
            np_img = image
        
        # If the image is in channels-first format (e.g., (3, H, W)), transpose it.
        if len(np_img.shape) == 3 and np_img.shape[0] == 3:
            np_img = np.transpose(np_img, (1, 2, 0))
        
        # --- Color Histogram Features ---
        # If the image is color (i.e. channels-last and 3 channels), compute per-channel histograms.
        if len(np_img.shape) == 3 and np_img.shape[-1] == 3:
            color_hist_features = []
            # Using 32 bins per channel
            for channel in range(3):
                channel_data = np_img[:, :, channel]
                hist, _ = np.histogram(channel_data, bins=32, range=(0, 256))
                hist = hist.astype("float")
                hist /= (hist.sum() + 1e-7)
                color_hist_features.extend(hist)
            color_hist_features = np.array(color_hist_features)
        else:
            color_hist_features = np.array([])
        
        # --- Grayscale Conversion ---
        # For the rest of the features, work with a grayscale image.
        if len(np_img.shape) == 3:
            if np_img.shape[-1] == 3:
                gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
            elif np_img.shape[-1] == 4:
                gray = cv2.cvtColor(np_img, cv2.COLOR_RGBA2GRAY)
            else:
                gray = np_img[:, :, 0]
        else:
            gray = np_img
        
        # Convert to uint8 for consistency in further processing.
        gray_uint8 = gray.astype(np.uint8)

        # --- HOG Features ---
        hog_features = hog(
            gray_uint8,
            orientations=hog_orientations,
            pixels_per_cell=hog_pixels_per_cell,
            cells_per_block=hog_cells_per_block,
            block_norm='L2-Hys',
            transform_sqrt=True,
            feature_vector=True
        )
        
        # --- LBP Features ---
        lbp = local_binary_pattern(gray_uint8, lbp_P, lbp_R, method=lbp_method)
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=lbp_n_bins, range=(0, lbp_n_bins))
        lbp_hist = lbp_hist.astype("float")
        lbp_hist /= (lbp_hist.sum() + 1e-7)
        
        # --- Wavelet Features ---
        wavelet_feats = extract_wavelet_features(gray_uint8, wavelet=wavelet, level=wavelet_level)
        
        # --- Hu Moments ---
        moments = cv2.moments(gray_uint8)
        huMoments = cv2.HuMoments(moments).flatten()
        # Log-transform for numerical stability.
        huMoments = -np.sign(huMoments) * np.log10(np.abs(huMoments) + 1e-7)
        
        # Concatenate all feature sets.
        combined_features = np.hstack([
            hog_features,
            lbp_hist,
            wavelet_feats,
            color_hist_features,
            huMoments
        ])
        X_chunk.append(combined_features)
        y_chunk.append(label)
    
    return np.array(X_chunk), np.array(y_chunk)

def extract_features_parallel(dataset, num_jobs=4, **kwargs):
    """
    Splits the dataset into index chunks and processes them in parallel.
    
    Parameters:
        dataset: A custom dataset supporting __getitem__(index).
        num_jobs: Number of parallel jobs.
        **kwargs: Additional parameters for feature extraction (HOG, LBP, wavelet, etc.).
    
    Returns:
        X_final: Numpy array of feature vectors.
        y_final: Numpy array of labels.
    """
    n = len(dataset)
    indices = list(range(n))
    # Split indices into roughly equal chunks.
    chunk_size = (n + num_jobs - 1) // num_jobs  # Ceiling division
    chunks = [indices[i:i + chunk_size] for i in range(0, n, chunk_size)]
    
    X_list = []
    y_list = []
    
    with ProcessPoolExecutor(max_workers=num_jobs) as executor:
        futures = [executor.submit(process_indices, dataset, chunk, **kwargs) for chunk in chunks]
        for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
            X_chunk, y_chunk = future.result()
            X_list.append(X_chunk)
            y_list.append(y_chunk)
    
    X_final = np.concatenate(X_list, axis=0)
    y_final = np.concatenate(y_list, axis=0)
    
    return X_final, y_final

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
        fraction=args.fraction,
    )
    valid_dataset = DeepFakeDataset(
        csv_file=os.path.join(args.csv_dir, "valid.csv"),
        root_dir=args.data_dir,
        transform=transform,
        fraction=args.fraction,
    )
    test_dataset = DeepFakeDataset(
        csv_file=os.path.join(args.csv_dir, "test.csv"),
        root_dir=args.data_dir,
        transform=transform,
        fraction=args.fraction,
    )

    # Extração de features (flatten das imagens)
    
    cache_path = "./cache"
    
    if not os.path.exists(cache_path):
        os.makedirs(cache_path)
    
    if os.path.exists(os.path.join(cache_path, "X_train.npy")):
        logging.info("Carregando features extraídas anteriormente")
        X_train = np.load(os.path.join(cache_path, "X_train.npy"))
        y_train = np.load(os.path.join(cache_path, "y_train.npy"))
        X_valid = np.load(os.path.join(cache_path, "X_valid.npy"))
        y_valid = np.load(os.path.join(cache_path, "y_valid.npy"))
        X_test = np.load(os.path.join(cache_path, "X_test.npy"))
        y_test = np.load(os.path.join(cache_path, "y_test.npy"))
    else:
        # Extração de features (HOG + LBP) em paralelo
        logging.info("Extraindo features do conjunto de treino")
        X_train, y_train = extract_features_parallel(train_dataset)
        logging.info("Extraindo features do conjunto de validação")
        X_valid, y_valid = extract_features_parallel(valid_dataset)
        logging.info("Extraindo features do conjunto de teste")
        X_test, y_test = extract_features_parallel(test_dataset)
        
        # Salva as features extraídas
        np.save(os.path.join(cache_path, "X_train.npy"), X_train)
        np.save(os.path.join(cache_path, "y_train.npy"), y_train)
        np.save(os.path.join(cache_path, "X_valid.npy"), X_valid)
        np.save(os.path.join(cache_path, "y_valid.npy"), y_valid)
        np.save(os.path.join(cache_path, "X_test.npy"), X_test)
        np.save(os.path.join(cache_path, "y_test.npy"), y_test)
        
    # Normaliza os dados min max
    
    _min = np.min(X_train, axis=0)
    _max = np.max(X_train, axis=0)
    
    print(_min, _min.shape)
    print(_max, _max.shape)
    
    X_train = (X_train - _min) / (_max - _min + 1e-7)
    X_valid = (X_valid - _min) / (_max - _min + 1e-7)
    X_test = (X_test - _min) / (_max - _min + 1e-7)
    

    # Instancia o modelo Decision Tree
    # dt_model = DT(
    #     max_depth=None,
    #     random_state=args.seed,
    #     min_samples_split=10,
    #     min_samples_leaf=4,
    # )
    # dt_model = GTB(random_state=args.seed)

    dt_model = XGB(
        # n_estimators=300,
        # learning_rate=0.05,
        # max_depth=200,
        # random_state=args.seed,
        # use_label_encoder=True,
        # eval_metric='logloss'
    )
    
    # Otimização dos hiperparâmetros via GridSearchCV
    # param_grid = {
    #     'n_estimators': [100, 200, 300],
    #     'max_depth': [5, 10, 15],
    #     'learning_rate': [0.01, 0.1, 0.2],
    # }
    
    # logging.info("Realizando tuning dos hiperparâmetros com GridSearchCV")
    # best_params, best_score = dt_model.tune_parameters(param_grid, X_train, y_train, cv=2)
    # logging.info(f"Melhores parâmetros: {best_params} com acurácia de validação: {best_score:.2f}")

    # Treina o modelo com o conjunto completo de treino
    dt_model.fit(X_train, y_train)
    train_acc = dt_model.score(X_train, y_train)
    logging.info(f"Acurácia no treino: {train_acc * 100:.2f}%")
    # Avalia o modelo nos conjuntos de validação e teste
    valid_acc = dt_model.score(X_valid, y_valid)
    logging.info(f"Acurácia na validação: {valid_acc * 100:.2f}%")
    test_acc = dt_model.score(X_test, y_test)
    logging.info(f"Acurácia no teste: {test_acc * 100:.2f}%")
    
    dt_model.print_confusion_matrix(X_train, y_train)
    dt_model.print_confusion_matrix(X_valid, y_valid)
    dt_model.print_confusion_matrix(X_test, y_test)
    
    
    
    
    # Salva o modelo treinado
    joblib.dump(dt_model, args.save_model)
    logging.info(f"Modelo salvo em {args.save_model}")

if __name__ == "__main__":
    main()
