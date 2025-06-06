import torch
import torchvision.transforms as transforms
from torchvision import datasets
import os

# Hyperparameters matching dcgan_train_cifar10.py
img_size = 32
channels = 3
base_dir = "."

# Create directory for real dataset
os.makedirs(os.path.join(base_dir, "fixedDatasets"), exist_ok=True)

# Define transformations (same as in dcgan_train_cifar10.py)
transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),  # Normalize to [-1, 1]
])

# Load CIFAR-10 training dataset
cifar10 = datasets.CIFAR10(
    root="data/cifar10",
    train=True,
    download=True,
    transform=transform
)

# Convert dataset to a single tensor
real_samples = torch.stack([cifar10[i][0] for i in range(len(cifar10))])

# Save as .pt file
real_dir = os.path.join(base_dir, "fixedDatasets", "cifar10.pt")
torch.save(real_samples, real_dir)
print(f"Saved {real_samples.size(0)} real CIFAR-10 samples to {real_dir}")