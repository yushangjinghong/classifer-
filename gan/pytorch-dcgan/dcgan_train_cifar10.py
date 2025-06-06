import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets
from torch.autograd import Variable

# --- 条件生成器 (Conditional Generator) ---
class Generator(nn.Module):
    def __init__(self, latent_dim=100, num_classes=10, img_shape=(3, 32, 32)):
        super(Generator, self).__init__()
        self.img_shape = img_shape
        self.num_classes = num_classes

        # 嵌入层，将类别标签转换为向量
        self.label_emb = nn.Embedding(num_classes, num_classes) # 将类别转换为10维向量

        self.init_size = img_shape[1] // 4  # 初始大小 8x8
        # 输入维度是 潜变量维度 + 类别嵌入维度
        self.l1 = nn.Linear(latent_dim + num_classes, 128 * self.init_size ** 2)
        
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),  # 上采样到 16x16
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),  # 上采样到 32x32
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, img_shape[0], 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z, labels):
        # 将标签嵌入并与噪声向量拼接
        # labels 转换为 one-hot 编码可能更常见，这里直接使用 Embedding
        # 也可以考虑将 label_emb 的输出展平后与 z 拼接
        
        # 将标签转换为 one-hot 向量并与 z 拼接
        # label_onehot = torch.nn.functional.one_hot(labels, num_classes=self.num_classes).float()
        # gen_input = torch.cat((z.view(z.size(0), -1), label_onehot), -1)

        # 另一种更常用的方式是使用Embedding，并直接将Embedding结果与噪声拼接
        gen_input = torch.cat((self.label_emb(labels), z.view(z.size(0), -1)), -1)

        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

# --- 条件判别器 (Conditional Discriminator) ---
class Discriminator(nn.Module):
    def __init__(self, num_classes=10, img_shape=(3, 32, 32)):
        super(Discriminator, self).__init__()
        self.img_shape = img_shape
        self.num_classes = num_classes

        # 嵌入层，将类别标签转换为向量
        self.label_emb = nn.Embedding(num_classes, num_classes) # 可以与图像大小匹配，或拼接后通过卷积处理

        # 判别器也需要处理图像和标签
        # 一种常见做法是将标签信息也转换为与图像尺寸相同，并与图像通道拼接
        # 或者在更深的层进行拼接，取决于具体设计

        # 这里选择将标签嵌入转换为与图像通道数相匹配的尺寸，然后与图像拼接
        # 然后通过卷积层进行处理
        
        # 为了简洁和保持DCGAN结构，我们先将标签嵌入信息和图像信息在第一个卷积层输入前进行融合
        # 或者在 Flatten 之后，与图像特征一起输入到最终的 Linear 层

        # 推荐做法是将标签信息转换成与图像形状匹配的特征图，然后与图像拼接
        # 比如，将 label_emb(labels) 转换为 (batch_size, num_classes, img_size, img_size)
        # 并与 img_shape[0] + num_classes 作为第一个卷积层的输入通道数

        self.model = nn.Sequential(
            # 输入通道数是 (图像通道数 + 类别数)，因为我们将标签嵌入信息与图像通道拼接
            nn.Conv2d(img_shape[0] + num_classes, 64, 3, stride=2, padding=1), # 假设标签信息也作为通道输入
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1), # 最终输出一个标量表示真伪
            nn.Sigmoid(),
        )

    def forward(self, img, labels):
        # 将标签嵌入并扩充到图像的尺寸，然后与图像通道拼接
        # labels 转换为 one-hot 编码
        label_onehot = torch.nn.functional.one_hot(labels, num_classes=self.num_classes).float()
        # 扩充 one-hot 向量的维度使其能与图像拼接
        label_onehot = label_onehot.view(label_onehot.size(0), self.num_classes, 1, 1)
        label_onehot = label_onehot.repeat(1, 1, self.img_shape[1], self.img_shape[2])
        
        # 将标签信息和图像拼接作为判别器的输入
        d_input = torch.cat((img, label_onehot), 1) # 在通道维度拼接

        validity = self.model(d_input)
        return validity

# --- 超参数 (添加类别数) ---
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of second order momentum")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--num_classes", type=int, default=10, help="number of classes for dataset") # 添加类别数
parser.add_argument("--sample_interval", type=int, default=500, help="interval between image sampling") # 保存图片间隔
parser.add_argument("--resume_epoch", type=int, default=None, help="resume training from this epoch")
opt = parser.parse_args()

img_shape = (opt.channels, opt.img_size, opt.img_size)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 损失函数
adversarial_loss = nn.BCELoss()

# 初始化生成器和判别器
generator = Generator(latent_dim=opt.latent_dim, num_classes=opt.num_classes, img_shape=img_shape).to(device)
discriminator = Discriminator(num_classes=opt.num_classes, img_shape=img_shape).to(device)

# 优化器
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# --- 恢复训练状态 (需要更新以适应新的模型输入) ---
start_epoch = 0
if opt.resume_epoch is not None:
    resume_epoch = opt.resume_epoch - 1  # 加载前一个 epoch 的状态
    checkpoint_path = f"saved_models/checkpoint_epoch_{resume_epoch}.pth"
    generator_path = f"saved_models/generator_epoch_{resume_epoch}.pth" # 旧的单独保存的生成器
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device) # 添加 map_location
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        start_epoch = opt.resume_epoch
        print(f"Resumed training from checkpoint_epoch_{resume_epoch}.pth")
    elif os.path.exists(generator_path):
        # 如果只存在旧的生成器模型，则只加载生成器，判别器和优化器会重新初始化
        print("Warning: Only generator checkpoint found. Discriminator and optimizers will be re-initialized.")
        generator.load_state_dict(torch.load(generator_path, map_location=device))
        start_epoch = opt.resume_epoch
    else:
        raise FileNotFoundError(f"Neither {checkpoint_path} nor {generator_path} exists for resuming.")


# --- 数据加载器 (CIFAR-10 数据集，现在我们也要用到标签) ---
os.makedirs("data", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.CIFAR10(
        root="data/cifar10",
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.Resize(opt.img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)

# --- 训练循环 (修改以处理标签) ---
os.makedirs("saved_models", exist_ok=True)
os.makedirs("generated_images", exist_ok=True)

for epoch in range(start_epoch, opt.n_epochs):
    for i, (imgs, labels) in enumerate(dataloader): # 现在dataloader会返回图像和标签
        # 真实标签和假标签
        valid = Variable(torch.ones(imgs.size(0), 1).to(device), requires_grad=False)
        fake = Variable(torch.zeros(imgs.size(0), 1).to(device), requires_grad=False)

        # 将真实图像和标签移到设备
        real_imgs = imgs.to(device)
        labels = labels.to(device) # 将标签也移到设备

        # -----------------
        #  训练生成器
        # -----------------
        optimizer_G.zero_grad()
        
        # 生成随机噪声向量
        z = Variable(torch.randn(imgs.size(0), opt.latent_dim, 1, 1).to(device))
        # 随机选择类别标签用于生成器
        # 为什么是随机选择？因为生成器在训练时需要尝试生成所有类别的图像
        gen_labels = Variable(torch.randint(0, opt.num_classes, (imgs.size(0),)).to(device))
        
        # 生成器生成图像时传入噪声和标签
        gen_imgs = generator(z, gen_labels)
        
        # 判别器判断生成图像和其对应的标签
        g_loss = adversarial_loss(discriminator(gen_imgs, gen_labels), valid) # 判别器也要知道是哪个标签
        
        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  训练判别器
        # ---------------------
        optimizer_D.zero_grad()
        
        # 判别器判断真实图像和真实标签
        real_loss = adversarial_loss(discriminator(real_imgs, labels), valid)
        
        # 判别器判断生成图像和生成标签 (这里很重要，使用 detach() 防止梯度回传到生成器)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach(), gen_labels), fake)
        
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

        print(f"[Epoch {epoch}/{opt.n_epochs}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")

        # 每隔 sample_interval 批次保存一次生成图像
        if i % opt.sample_interval == 0:
            # 在生成测试图像时，我们希望生成所有类别的图像
            # 为每个类别生成10张图像，以便查看效果 (总共100张)
            
            # 创建固定噪声，以便在不同epoch下生成相同的图像，便于观察生成质量的提升
            if i == 0 and epoch == start_epoch: # 仅在第一次运行时创建固定噪声
                fixed_noise = Variable(torch.randn(opt.num_classes * 10, opt.latent_dim, 1, 1)).to(device)
                # 为这些图像创建对应的标签，例如：0,0,..,0 (10张), 1,1,..,1 (10张), ..., 9,9,..,9 (10张)
                fixed_labels = Variable(torch.cat([torch.full((10,), c, dtype=torch.long) for c in range(opt.num_classes)])).to(device)
            
            # 使用固定噪声和固定标签生成图像
            gen_imgs_sample = generator(fixed_noise, fixed_labels)
            save_image(gen_imgs_sample.data[:100], f"generated_images/epoch_{epoch}_batch_{i}.png", nrow=10, normalize=True)


    # 每个 epoch 保存检查点（包括生成器、判别器、优化器状态）
    checkpoint = {
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_state_dict': optimizer_D.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, f"saved_models/checkpoint_epoch_{epoch}.pth")
    # 兼容旧代码，单独保存生成器权重 (现在生成器是带标签的，保存它更有意义)
    torch.save(generator.state_dict(), f"saved_models/generator_epoch_{epoch}.pth")