import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
from arch.deepfake_cnn import CNN
from arch.adversarial_generator import MidTermGenerator
from dataloader.dataloader import DeepFakeDataset, DataLoader
from torchvision import transforms
import sys

class GradCAM:
    def __init__(self, model, target_layer):
        """
        model: The trained CNN model (nn.Module)
        target_layer: The specific layer (module) from which we extract gradients/features,
                      e.g., model.conv4 or model.batchnorm4, etc.
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # -- Register hooks on the chosen layer --
        self.forward_hook = target_layer.register_forward_hook(self.save_activation)
        self.backward_hook = target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        """Hook to save the forward feature maps (activations)."""
        self.activations = output.detach()

    def save_gradient(self, module, grad_in, grad_out):
        """Hook to save the gradients of the target layer."""
        # grad_out is a tuple; we want the gradient w.r.t. the outputs
        self.gradients = grad_out[0].detach()

    def __call__(self, x, class_idx=None):
        """
        x: Input image tensor of shape (B, C, H, W).
        class_idx: (Optional) Target class index to compute Grad-CAM for.
                   If None, we use the predicted class from the model.
        Returns: heatmap (H, W) as a numpy array of normalized Grad-CAM heat.
        """
        # Forward pass
        logits = self.model(x)

        # If class_idx is not specified, take the highest scoring class
        if class_idx is None:
            class_idx = torch.argmax(logits, dim=1).item()  # integer

        # Zero all existing gradients
        self.model.zero_grad()

        # Extract the logit corresponding to our class of interest
        score = logits[:, class_idx]  # shape: (B,)

        # Backprop to get gradients
        score.backward(retain_graph=True)

        # Grab the gradients and activations
        gradients = self.gradients  # shape: (B, C, H', W')
        activations = self.activations  # shape: (B, C, H', W')

        # For simplicity, assume batch size = 1. Otherwise, index accordingly.
        b, k, u, v = gradients.shape

        # 1) Global-average-pool the gradients to get weights per channel
        alpha = gradients.view(b, k, -1).mean(2)  # shape: (B, C)

        # 2) Weight each channel in the activations by alpha
        # Expand alpha to match the spatial dims of the activations
        alpha = alpha.view(b, k, 1, 1)
        weighted_activations = alpha * activations  # shape: (B, C, H', W')

        # 3) Sum across channels to get a single heatmap
        heatmap = weighted_activations.sum(dim=1).squeeze(0)  # shape: (H', W')

        # 4) Apply a ReLU (as in the Grad-CAM paper)
        heatmap = F.relu(heatmap)

        # 5) Normalize the heatmap for visualization (0 to 1)
        heatmap -= heatmap.min()
        heatmap /= heatmap.max() if heatmap.max() != 0 else 1e-8

        # Convert to numpy for easier plotting
        heatmap = heatmap.cpu().numpy()
        return heatmap

    def remove_hooks(self):
        """Remove the forward/backward hooks (to avoid potential memory leaks)."""
        self.forward_hook.remove()
        self.backward_hook.remove()


def show_gradcam_on_image(img_tensor, heatmap):
    """
    img_tensor: shape (1, 3, H, W) in [0,1] or [0,255], PyTorch Tensor.
    heatmap: shape (H', W'), a 2D numpy array from Grad-CAM.
             If H' != H, W' != W, we'll resize it to match.
    """
    # 1) Convert tensor to numpy with shape (H, W, 3)
    #    Assuming it's already scaled [0..1], but if not, scale accordingly.
    img_np = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    H, W = img_np.shape[:2]

    # 2) Resize heatmap to match the image size
    heatmap_resized = cv2.resize(heatmap, (W, H))

    # 3) Convert heatmap to a color map (jet, etc.)
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB) / 255.0

    # 4) Overlay the heatmap on the image (blend it)
    overlay = 0.5 * img_np + 0.5 * heatmap_colored
    overlay = np.clip(overlay, 0, 1)

    # 5) Plot side-by-side: original image, heatmap, overlay
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(img_np)
    axs[0].set_title("Original Image")
    axs[0].axis("off")

    axs[1].imshow(heatmap_resized, cmap="jet")
    axs[1].set_title("Grad-CAM Heatmap")
    axs[1].axis("off")

    axs[2].imshow(overlay)
    axs[2].set_title("Heatmap Overlay")
    axs[2].axis("off")

    plt.tight_layout()
    plt.show()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    idx = int(sys.argv[1]) if len(sys.argv) > 1 else np.random.randint(0, 1000)
    print(idx)
    
    # 1) Assume you have your trained model
    model = CNN()
    model.load_state_dict(torch.load("../results/deepfake/cnn_65_99.03.pth"))
    model.eval()  # inference mode
    model.to(device)

    # 2) Suppose you have one test image from your dataset or a custom image
    #    and you want a single batch of size=1
    image_tensor, label = (
        DeepFakeDataset(
            csv_file="../data/test.csv",
            root_dir="../data",
            transform=transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize((128, 128)),
                    transforms.ToTensor(),
                ]
            ),
       ).get_real_image(idx)
    )

    image_tensor = image_tensor.unsqueeze(0).to(device)  # shape (1, 3, H, W)

    # 3) Instantiate Grad-CAM on the last convolution block (conv4)
    gradcam = GradCAM(model, model.conv4)

    # 4) Run Grad-CAM
    heatmap = gradcam(image_tensor, class_idx=None)
    # If class_idx=None, it uses the predicted label.
    # If you specifically want the "fake" class (say label=1), pass class_idx=1.

    # 5) Visualize
    show_gradcam_on_image(image_tensor.cpu(), heatmap)
    
    # Same steps as above, but now for a different layer (conv3)
    gradcam = GradCAM(model, model.conv3)
    heatmap = gradcam(image_tensor, class_idx=None)
    show_gradcam_on_image(image_tensor.cpu(), heatmap)

    
    # Do gradcam with the image generated by the generator
    generator = MidTermGenerator()
    generator.load_state_dict(torch.load("../results/adversarial/good_perception/e01_allreal_adversary.pth"))
    generator.eval()
    generator.to(device)
    
    with torch.no_grad():
        adv_img = generator(image_tensor)
        

    gradcam.remove_hooks()


if __name__ == "__main__":
    main()
