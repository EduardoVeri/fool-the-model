import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from arch.adversarial_generator import MidTermGenerator


def visualize_adversarial_examples(device, test_data_loader, generator:MidTermGenerator, EPSILON):
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

        # Move tensors back to CPU and undo the [-1,1] normalization for display
        orig_img_vis = inv_transform(img)[0].detach().cpu()
        adv_img_vis = inv_transform(adv_img)[0].detach().cpu()

        # Compute the difference (still in [0,1] after inv_transform)
        diff_img = torch.abs(adv_img_vis - orig_img_vis)

        # Optionally, rescale the difference so the highest difference is 1.
        # This makes small changes more visible.
        if diff_img.max() > 0:
            diff_img = diff_img / diff_img.max()

        # Convert for plotting
        orig_img_vis = orig_img_vis.permute(1, 2, 0)
        adv_img_vis = adv_img_vis.permute(1, 2, 0)
        diff_img_vis = diff_img.permute(1, 2, 0)

        # Show original
        ax[0, count].imshow(orig_img_vis)
        if count % 2 == 0:
            ax[0, count].set_title("Real")
        else:
            ax[0, count].set_title("Fake")
        ax[0, count].axis("off")

        # Show adversarial
        ax[1, count].imshow(adv_img_vis)
        ax[1, count].set_title("Adversarial")
        ax[1, count].axis("off")

        # Show the difference (you can also add a cmap, e.g., cmap='bwr')
        ax[2, count].imshow(diff_img_vis, cmap="bwr")
        ax[2, count].set_title("Difference")
        ax[2, count].axis("off")

        count += 1
    plt.tight_layout()
    plt.show()
