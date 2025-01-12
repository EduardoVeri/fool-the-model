# Fool-the-Model

A framework for training and evaluating adversarial attacks against DeepFake classification models. This project provides:

- CNN for DeepFake detection
- Adversarial Generator to craft subtle perturbations
- Training scripts for both the detector and the adversarial generator

## Requirements

1. Python 3.12 or higher
   Ensure you have Python >= 3.12 installed. You can use system Python or a conda/mamba environment.

2. (Optional) Micromamba
   We recommend using micromamba (https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html) for faster environment management.

3. Install Dependencies
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Train the CNN classifier
   ```bash
   python deepfakes_train.py
   ```
   This trains a DeepFake detection CNN on your dataset.

2. Train the adversarial generator
   ```bash
   python adversary_train.py --train -e <experiment-name>
   ```
   This script generates perturbations that attempt to fool the classifier.

3. Adjust Hyperparameters
   Hyperparameters and configurations (e.g., learning rate, batch size) can be tweaked in the relevant scripts or YAML config files to suit your needs.

Visual Results

Below is an example of an original (real) image versus its adversarial counterpart produced by the generator. The adversarial image looks very similar to the original but is crafted to cause the DeepFake classifier to misclassify.

| Original                  | Adversarial                     |
|---------------------------|---------------------------------|
| ![Original Image](docs/original_example.jpg) | ![Adversarial Image](docs/adv_example.jpg) |

For a more extensive visualization, you can run a script (e.g., visualize_adversarial_examples.py) or integrate the images directly into notebooks or reports.

If you have any questions or issues, feel free to open an issue or submit a pull request. We welcome contributions to improve and expand this framework!
