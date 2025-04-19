from lightning.pytorch import LightningDataModule
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

class XRayDataModule(LightningDataModule):
    def __init__(self, data_dir, batch_size=32, num_workers=4, image_size=256): # Increased default num_workers
        super().__init__()
        self.save_hyperparameters()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size

        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((self.image_size, self.image_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        if self.train_dataset is not None and self.val_dataset is not None and self.test_dataset is not None:
            return

        full_dataset = datasets.ImageFolder(root=self.data_dir, transform=self.transform)
        class_to_idx = full_dataset.class_to_idx
        print(f"Class to index mapping: {class_to_idx}")

        # (70% train, 15% val, 15% test)
        total_size = len(full_dataset)
        train_size = int(0.7 * total_size)
        val_size = int(0.15 * total_size)
        test_size = total_size - train_size - val_size

        print(f"Total: {total_size}, Train: {train_size}, Val: {val_size}, Test: {test_size}")
        if train_size + val_size + test_size != total_size:
             print("Warning: Split sizes do not exactly match total size due to rounding.")
             train_size = total_size - val_size - test_size
             print(f"Adjusted Train size: {train_size}")

        if train_size < 0 or val_size < 0 or test_size < 0:
            raise ValueError("Calculated dataset split sizes are invalid.")

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(12947) # Seed for reproducible splits
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, persistent_workers=True if self.num_workers > 0 else False)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True if self.num_workers > 0 else False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True if self.num_workers > 0 else False)
