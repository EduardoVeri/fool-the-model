# Fool-the-Model

A framework for training and evaluating adversarial attacks against DeepFake classification models. This project provides:

- CNN, Unet and XGBoost classifiers to detect DeepFakes
- Adversarial Generator to craft subtle perturbations
- Training scripts for both the detector and the adversarial generator
- Visualization tools to inspect adversarial examples

## Contents
- [Fool-the-Model](#fool-the-model)
  - [Contents](#contents)
  - [Requirements](#requirements)
  - [Download Dataset](#download-dataset)
  - [Usage](#usage)
    - [Train the CNN or UNET classifier](#train-the-cnn-or-unet-classifier)
    - [Train the XGBoost classifier](#train-the-xgboost-classifier)
    - [Train the adversarial generator](#train-the-adversarial-generator)
    - [Visual Results](#visual-results)

## Requirements

1. (Optional) Micromamba
   We recommend using [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html) for faster environment management.

2. Python 3.12 or higher
   Ensure you have Python >= 3.12 installed. You can use system Python or a conda/mamba environment.

3. Install Dependencies
   ```bash
   pip install -r requirements.txt
   ```

It's recommended to execute the scripts in a computer with a GPU to speed up the training process and with enough memory to handle the dataset.

## Download Dataset

1. Download the DeepFake Detection Challenge dataset from Kaggle. [Click Here](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces) to access the dataset.
2. Unzip the dataset and place it in the `data` directory. Change the directory structure to the following:
   ```
   data
   ├── train
   ├── valid
   ├── test
   ├── train.csv
   ├── valid.csv
   └── test.csv
   ```
   The `train.csv`, `valid.csv`, and `test.csv` files contain the paths to the images and their corresponding labels. The `train`, `valid`, and `test` directories contain the images.


(Linux only) You can also use the `download_dataset.sh` script to download and extract the dataset. Run the following command:
```bash
./download_dataset.sh
```

## Usage

### Train the CNN or UNET classifier
   ```bash
   python deepfakes_train.py --model <model-name> -s <model-output-path> --train 
   ```
   This script trains a classifier to detect DeepFakes. You can choose between a CNN or UNET model.

### Train the XGBoost classifier
   ```bash
   python classical_deepfakes_train.py
   ```
   This script trains an XGBoost classifier to detect DeepFakes.

### Train the adversarial generator
   For training the adversarial generator using the CNN or UNET as victim, you can use the following command:
   
   ```bash
   python adversary_train.py --train -e <experiment-name>
   ```
   This script generates perturbations that attempt to fool the classifier. It uses a config file to specify the training parameters in `configs/adv_config.yaml`.

   For `XGBoost`, you can use the following command:
   ```bash
   python classical_adversary_train.py --train -e <experiment-name>
   ```

### Visual Results

Some scripts will automatically show visual results after training in a pop up window. We also provide a script to visualize the heatmap of the neural network's using the GradCAM technique. You can run the following command:
   ```bash
   python gradcam.py
   ```
   You can choose between the CNN and UNET models. XGBoost does not have a heatmap visualization.
