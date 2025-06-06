import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
import os

# DCGAN Generator (same as in dcgan_generate_cifar10.py)
class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_shape=(3, 32, 32)):
        super(Generator, self).__init__()
        self.img_shape = img_shape
        self.init_size = img_shape[1] // 4  # Initial size 8x8
        self.l1 = nn.Linear(latent_dim, 128 * self.init_size ** 2)
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),  # Upsample to 16x16
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),  # Upsample to 32x32
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, img_shape[0], 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        z = z.view(z.size(0), -1)
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
latent_dim = 100
img_size = 32
channels = 3
img_shape = (channels, img_size, img_size)
num_samples = 50000  # Generate 50,000 fake images to match CIFAR-10 size
batch_size = 128
base_dir = "."

# Load generator
generator = Generator(latent_dim=latent_dim, img_shape=img_shape).to(device)
generator.load_state_dict(torch.load("saved_models/generator_epoch_199.pth", weights_only=True))
generator.eval()

# Create directory for fake images
os.makedirs(os.path.join(base_dir, "cifar10", "savedGenImages"), exist_ok=True)

# Generate fake images in batches
fake_samples = []
with torch.no_grad():
    for i in range(0, num_samples, batch_size):
        batch_num = min(batch_size, num_samples - i)
        noise = torch.randn(batch_num, latent_dim, 1, 1).to(device)
        gen_imgs = generator(noise)
        fake_samples.append(gen_imgs.cpu())
        print(f"Generated batch {i // batch_size + 1}/{(num_samples + batch_size - 1) // batch_size}")

# Concatenate all fake images
fake_samples = torch.cat(fake_samples, dim=0)

# Save fake images
fake_dir = os.path.join(base_dir, "cifar10", "savedGenImages", "cifar10_g32_d32_ep199_generator.pt")
torch.save(fake_samples, fake_dir)
print(f"Saved {fake_samples.size(0)} fake CIFAR-10 samples to {fake_dir}")

# Save a sample collage for visualization
save_image(fake_samples[:64], "generated_images/fake_cifar10_collage.png", nrow=8, normalize=True)
print("Saved sample collage to generated_images/fake_cifar10_collage.png")