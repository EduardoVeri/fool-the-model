from torch.utils.data import Dataset, DataLoader
import os
from skimage import io
import pandas as pd


class DeepFakeDataset(Dataset):
    def __init__(self, csv_file, root_dir, fraction=0.01, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        if fraction < 1:
            # Make a sample of the dataset
            # The values of Fake and Real must be balanced
            fake_images = self.data[self.data["label"] == 0]
            real_images = self.data[self.data["label"] == 1]
            fake_images = fake_images.sample(frac=fraction / 2)
            real_images = real_images.sample(frac=fraction / 2)
            self.data = pd.concat([fake_images, real_images])
            
        self.data = self.data.reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.__open_image(idx)
        label = self.data.iloc[idx]["label"]

        if self.transform:
            image = self.transform(image)

        return image, label

    def __open_image(self, idx):
        img_path = os.path.join(self.root_dir, self.data.iloc[idx]["path"])
        image = io.imread(img_path)
        return image
    
    def get_fake_image(self, idx):
        fake_images = self.data[self.data["label"] == 0].reset_index(drop=True)
        fake_images.sort_values("path", inplace=True)
        image = self.__open_image(fake_images.iloc[idx].name)
        
        if self.transform:
            image = self.transform(image)
        
        return image, 0

    def get_real_image(self, idx):
        real_images = self.data[self.data["label"] == 1].reset_index(drop=True).sort_values("path")
        image = self.__open_image(real_images.iloc[idx].name)
        
        if self.transform:
            image = self.transform(image)
        
        return image, 1
    

def get_dataloaders(path: str, batch_size: int, transform: object, fraction: float = 1):
    train_dataset = DeepFakeDataset(
        csv_file=os.path.join(path, "train.csv"),
        root_dir=path,
        fraction=fraction,
        transform=transform,
    )
    valid_dataset = DeepFakeDataset(
        csv_file=os.path.join(path, "valid.csv"),
        root_dir=path,
        fraction=fraction,
        transform=transform,
    )
    test_dataset = DeepFakeDataset(
        csv_file=os.path.join(path, "test.csv"),
        root_dir=path,
        fraction=fraction,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, valid_loader, test_loader
