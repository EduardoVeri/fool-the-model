# Fool-the-Model

This repository contains a framework for training and evaluating adversarial attacks against DeepFake classification models. It includes:

- A CNN for DeepFake detection.
- An adversarial generator to craft subtle perturbations.
- Scripts to train both the detector and the adversarial generator.

## Requirements

To run the project you will need:

1. Python 3.12 or higher installed in your system or conda/mamba environment.

2. (Optional) We recommend installing the Micromamba package manager. It is a faster and more efficient alternative to conda. Follow the steps in the [official site](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html) to install it.

3. Install requirements:  
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Ensure you have a properly trained CNN by running:
   ```bash
   python deepfakes_train.py
   ```
2. Train the adversarial generator (or use the pre-trained one):
   ```bash
   python adversary_train.py --train -e <experiment-name>
   ```
3. Adjust configurations and hyperparameters in the YAML config if needed.

For more details, explore the scripts in this repository.
