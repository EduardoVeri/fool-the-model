import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm
from .utils import CNN, DeepFake
from torch.utils.data import DataLoader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_size = 128
img_channels = 3
batch_size = 16
lr = 0.0002
num_epochs = 100
lambda_perceptual = 0.1


class Generator(nn.Module):
    def __init__(self, img_channels):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(img_channels, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, img_channels, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        perturbation = self.model(x)
        adv_x = x + perturbation
        adv_x = torch.clamp(adv_x, 0, 1)
        return adv_x


classifier = CNN().to(device)
classifier.load_state_dict(torch.load("path_to_your_classifier.pth"))
classifier.eval()


generator = Generator(
    input_dim=img_size, img_channels=img_channels, img_size=img_size
).to(device)

optimizer_g = optim.Adam(generator.parameters(), lr=lr)

adversarial_loss = nn.CrossEntropyLoss() 
perceptual_loss = nn.MSELoss()

transform = transforms.Compose(
    [transforms.Resize((img_size, img_size)), transforms.ToTensor()]
)

csv_file = "../data/140k-real-and-fake-faces/train.csv"
root_dir = "../data/140k-real-and-fake-faces/train"
dataset = DeepFake(csv_file=csv_file, root_dir=root_dir, transform=transform)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train(generator, classifier, data_loader):
    for epoch in range(num_epochs):
        for imgs, labels in tqdm(
            data_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False
        ):
            imgs = imgs.to(device)
            labels = labels.to(device)

            adv_imgs = generator(imgs)

            outputs = classifier(adv_imgs)
            loss_adv = -adversarial_loss(outputs, labels)

            victim_acc = (outputs.argmax(1) == labels).float().mean()

            loss_perceptual = perceptual_loss(adv_imgs, imgs)

            total_loss = loss_adv + lambda_perceptual * loss_perceptual

            optimizer_g.zero_grad()
            total_loss.backward()
            optimizer_g.step()

            print(
                f"Epoch [{epoch+1}/{num_epochs}] - "
                f"Deception Loss: {loss_adv.item():.4f}, "
                f"Perceptual Loss: {loss_perceptual.item():.4f}"
                f"Victim Accuracy: {victim_acc.item():.4f}"
            )


if __name__ == "__main__":
    train(generator, classifier, data_loader)

    torch.save(generator.state_dict(), "adversarial_generator.pth")
