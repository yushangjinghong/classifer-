import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
import os

# --- Conditional DCGAN 生成器（与训练代码中的Conditional Generator一致） ---
class Generator(nn.Module):
    def __init__(self, latent_dim=100, num_classes=10, img_shape=(3, 32, 32)):
        super(Generator, self).__init__()
        self.img_shape = img_shape
        self.num_classes = num_classes

        # 嵌入层，将类别标签转换为向量
        self.label_emb = nn.Embedding(num_classes, num_classes)

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
        gen_input = torch.cat((self.label_emb(labels), z.view(z.size(0), -1)), -1)

        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

# --- 配置参数 ---
# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 超参数（确保与训练时的Conditional DCGAN参数一致）
latent_dim = 100
img_size = 32
channels = 3
num_classes = 10 # CIFAR-10 有 10 个类别
img_shape = (channels, img_size, img_size)

# 定义 CIFAR-10 类别名称（可选，用于更好的可视化）
cifar10_classes = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# --- 加载 Conditional Generator ---
# 注意：这里我们加载的是Conditional Generator，所以需要传入 num_classes
generator = Generator(latent_dim=latent_dim, num_classes=num_classes, img_shape=img_shape).to(device)

# 确保加载的是 Conditional DCGAN 训练后的权重文件
# 假设您已经训练了Conditional DCGAN，并且保存了checkpoint
# 如果您保存的是整个 checkpoint，需要这样加载：
checkpoint_path = "saved_models/checkpoint_epoch_199.pth" # 替换为您的实际路径和 epoch
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    print(f"Loaded Conditional Generator from {checkpoint_path}")
elif os.path.exists("saved_models/generator_epoch_199.pth"):
    # 如果您只保存了生成器权重（仅在旧版本或调试时），请确保它是Conditional Generator的结构
    print("Warning: Loading generator_epoch_199.pth. Please ensure this is from a trained Conditional DCGAN.")
    generator.load_state_dict(torch.load("saved_models/generator_epoch_199.pth", map_location=device))
else:
    raise FileNotFoundError(f"No checkpoint found at {checkpoint_path} or saved_models/generator_epoch_199.pth. "
                            "Please ensure you have trained a Conditional DCGAN and saved its weights.")

generator.eval() # 将模型设置为评估模式，关闭 dropout/batchnorm 更新

# --- 生成指定标签的图像 ---
# 目标：为每个类别生成一些图像
num_images_per_class = 5 # 每个类别生成 5 张图片
total_samples = num_images_per_class * num_classes

# 生成随机噪声
# 注意：Conditional Generator的 forward 方法不再是 (batch_size, latent_dim, 1, 1)
# 而是 (batch_size, latent_dim) 或 (batch_size, latent_dim, 1, 1) 取决于你的l1层如何处理
# 我们的 Generator 内部会展平 z，所以这里可以保持 (batch_size, latent_dim, 1, 1)
noise = torch.randn(total_samples, latent_dim, 1, 1).to(device)

# 创建对应的标签（例如，0,0,0,0,0, 1,1,1,1,1, ... , 9,9,9,9,9）
labels = torch.cat([torch.full((num_images_per_class,), c, dtype=torch.long) for c in range(num_classes)]).to(device)

# 生成图像
with torch.no_grad():
    gen_imgs = generator(noise, labels) # 传入噪声和标签

# --- 保存和可视化图像 ---
output_dir = "generated_conditional_images"
os.makedirs(output_dir, exist_ok=True)

# 保存生成的图像为 .pt 格式
output_tensor_path = os.path.join(output_dir, "conditional_samples_cifar10.pt")

# Assuming 'fixed_labels' is the tensor containing the class labels used for generation.
# Make sure fixed_labels is defined and accessible at this point in your code.
torch.save({
    'images': gen_imgs.cpu(),  # Save the generated images
    'labels': labels.cpu() # Save the corresponding labels
}, output_tensor_path)

# Update the print statement to reflect that it's now a dictionary
print(f"Generated images (with labels) saved as dictionary to {output_tensor_path}")

# 保存生成的图像为 PNG
# 为了更好地可视化，每行显示一个类别的图片
save_image(gen_imgs, os.path.join(output_dir, "conditional_samples_cifar10.png"),
           nrow=num_images_per_class, normalize=True,
           # 添加 pad_value 来增加类别之间的间隔，使其更清晰
           padding=2, pad_value=0.5 # 0.5 是一个中灰色，用于分隔
)

# 可视化生成的图像
gen_imgs_display = gen_imgs.cpu().numpy()
gen_imgs_display = 0.5 * gen_imgs_display + 0.5  # 反归一化到 [0, 1]

fig, axes = plt.subplots(num_classes, num_images_per_class, figsize=(num_images_per_class * 2, num_classes * 2))

for c in range(num_classes):
    for i in range(num_images_per_class):
        idx = c * num_images_per_class + i
        ax = axes[c, i]
        ax.imshow(np.transpose(gen_imgs_display[idx], (1, 2, 0)))
        ax.axis("off")
        if i == 0: # 在每行的第一个图像旁边标注类别
            ax.set_title(cifar10_classes[c], loc='left', color='blue', fontsize=10)

plt.suptitle("Generated Images by Class (Conditional GAN)", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 调整布局以避免标题重叠
plt.show()