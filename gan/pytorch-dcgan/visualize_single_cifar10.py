import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os

# 设置与训练代码一致的变换
transform = transforms.Compose([
    transforms.Resize(32),  # 确保图像大小为 32x32
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize([0.5], [0.5]),  # 归一化到 [-1, 1]
])

# 加载 CIFAR-10 训练集
dataset = torchvision.datasets.CIFAR10(
    root="/root/autodl-tmp/data/cifar10",
    train=True,
    download=True,
    transform=transform
)

# CIFAR-10 的类别标签
classes = ('airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# 选择一张图片（以下提供三种方式，选择一种）
# 方式 1：随机选择一张图片
image_idx = np.random.randint(0, len(dataset))
image, label = dataset[image_idx]

# 方式 2：按索引选择（例如，第 0 张图片）
# image_idx = 0
# image, label = dataset[image_idx]

# 方式 3：按类别选择（例如，第一张猫的图片）
# target_class = 'cat'
# class_idx = classes.index(target_class)
# for idx in range(len(dataset)):
#     _, label = dataset[idx]
#     if label == class_idx:
#         image_idx = idx
#         image, label = dataset[image_idx]
#         break

# 反归一化以便显示
image = image.numpy()  # 转换为 numpy 数组
image = 0.5 * image + 0.5  # 反归一化到 [0, 1]

# 显示图片
plt.figure(figsize=(4, 4))
plt.imshow(np.transpose(image, (1, 2, 0)))  # 将 (C, H, W) 转换为 (H, W, C)
plt.title(f"Class: {classes[label]} (Index: {image_idx})", fontsize=12)
plt.axis('off')

# 保存图片
os.makedirs("/root/autodl-tmp/cifar10_images", exist_ok=True)
plt.savefig(f"/root/autodl-tmp/cifar10_images/single_image_idx_{image_idx}.png")
plt.show()