import numpy as np
import os

# 定义你想要生成的样本数量
sample_sizes = [25, 50, 100, 500, 1000, 5000, 10000]

# 创建一个目录来存储生成的数据，如果它不存在的话
output_dir = 'type_i_error_data'
os.makedirs(output_dir, exist_ok=True)

print("正在生成 I 型错误实验数据：\n")

for n_samples in sample_sizes:
    # 生成数据集 p：来自标准正态分布 N(0,1)
    p_type_i_error = np.random.randn(n_samples, 1)

    # 生成数据集 q：同样来自标准正态分布 N(0,1)
    q_type_i_error = np.random.randn(n_samples, 1)

    # 将数据集保存到 .npy 文件，文件名中包含样本数量
    p_filename = os.path.join(output_dir, f'type_i_error_p_normal_{n_samples}_samples.npy')
    q_filename = os.path.join(output_dir, f'type_i_error_q_normal_{n_samples}_samples.npy')

    np.save(p_filename, p_type_i_error)
    np.save(q_filename, q_type_i_error)

    print(f"  已为 {n_samples} 个样本生成数据：")
    print(f"    p_type_i_error 的形状: {p_type_i_error.shape}")
    print(f"    q_type_i_error 的形状: {q_type_i_error.shape}")
    print(f"    文件已保存到: {p_filename} 和 {q_filename}\n")

print("所有 I 型错误实验数据生成完毕。")