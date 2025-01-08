import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from arch.adversarial_generator import MidTermGenerator
import os
from os import path
from tqdm import tqdm


def visualize_adversarial_examples(
    device, test_data_loader, generator: MidTermGenerator
):
    generator.eval()
    fig, ax = plt.subplots(3, 5, figsize=(15, 8))
    inv_transform = transforms.Compose(
        [transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2])]
    )

    count = 0
    for i, (img, label) in enumerate(test_data_loader):
        if count == 5:
            break

        # Skip if the image is not a real image
        if i % 2 == 0 and label[0] == 0:
            continue

        # Skip if the image is not a deepfake
        if i % 2 == 1 and label[0] == 1:
            continue

        img = img.to(device)
        with torch.no_grad():
            adv_img = generator(img)


        orig_img_vis = inv_transform(img)[0].detach().cpu()
        adv_img_vis = inv_transform(adv_img)[0].detach().cpu()

        diff_img = torch.abs(adv_img_vis - orig_img_vis)

        if diff_img.max() > 0:
            diff_img = diff_img / diff_img.max()

        orig_img_vis = orig_img_vis.permute(1, 2, 0)
        adv_img_vis = adv_img_vis.permute(1, 2, 0)
        diff_img_vis = diff_img.permute(1, 2, 0)

        ax[0, count].imshow(orig_img_vis)
        if count % 2 == 0:
            ax[0, count].set_title("Real")
        else:
            ax[0, count].set_title("Fake")
        ax[0, count].axis("off")

        ax[1, count].imshow(adv_img_vis)
        ax[1, count].set_title("Adversarial")
        ax[1, count].axis("off")

        ax[2, count].imshow(diff_img_vis, cmap="bwr")
        ax[2, count].set_title("Difference")
        ax[2, count].axis("off")

        count += 1
    plt.tight_layout()
    plt.show()


def save_adversarial_examples(
    device, test_data_loader, generator: MidTermGenerator, output_folder: str
):
    
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(path.join(output_folder, "real"), exist_ok=True)
    os.makedirs(path.join(output_folder, "fake"), exist_ok=True)

    generator.eval()
    inv_transform = transforms.Compose(
        [transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2])]
    )

    for img, label in tqdm(test_data_loader, desc="Saving adversarial examples"):
        img = img.to(device)
        with torch.no_grad():
            adv_img = generator(img)

        for _img, _label, _adv_img in zip(img, label, adv_img):
            orig_img_vis = inv_transform(_img).detach().cpu()
            adv_img_vis = inv_transform(_adv_img).detach().cpu()

            diff_img = torch.abs(adv_img_vis - orig_img_vis)

            if diff_img.max() > 0:
                diff_img = diff_img / diff_img.max()

            orig_img_vis = orig_img_vis.permute(1, 2, 0)
            adv_img_vis = adv_img_vis.permute(1, 2, 0)
            diff_img_vis = diff_img.permute(1, 2, 0)

            fig, ax = plt.subplots(1, 3, figsize=(15, 8))

            ax[0].imshow(orig_img_vis)
            ax[0].set_title("Real" if _label == 1 else "Fake")
            ax[0].axis("off")

            ax[1].imshow(adv_img_vis)
            ax[1].set_title("Adversarial")
            ax[1].axis("off")

            ax[2].imshow(diff_img_vis, cmap="bwr")
            ax[2].set_title("Difference")
            ax[2].axis("off")

            folder = "real" if _label == 1 else "fake"
            plt.tight_layout()
            # generate a hash to prevent overwriting
            plt.savefig(path.join(output_folder, folder, f"{hash(_img)}.png"))

            plt.close(fig)
