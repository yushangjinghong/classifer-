import numpy as np
from scipy.stats import t
import os

# 定义输出目录
output_dir = 'gaussian_vs_t_data'
os.makedirs(output_dir, exist_ok=True)

print("开始生成高斯分布与学生 t 分布区分实验数据...\n")

# --- 情景一：固定样本数量 n=2000，变化自由度 nu ---
n_fixed_nu_vary = 2000
nu_values = [1, 3, 5, 10, 15, 20, 30] # 自由度 (nu) 列表

print(f"--- 情景一：固定样本数 n={n_fixed_nu_vary}，变化自由度 ---")
# 生成高斯分布数据 P (固定 n=2000)
p_gaussian_fixed_n = np.random.randn(n_fixed_nu_vary, 1)
p_gaussian_fixed_n_filename = os.path.join(output_dir, f'gaussian_vs_t_p_normal_n{n_fixed_nu_vary}.npy')
np.save(p_gaussian_fixed_n_filename, p_gaussian_fixed_n)
print(f"  已生成高斯分布样本: {p_gaussian_fixed_n_filename}")

for nu in nu_values:
    # 生成学生 t 分布数据 Q，并进行标准化（零均值，单位方差）
    # 注意：nu=1 时，t 分布是柯西分布，方差未定义。
    # 对于 nu <= 2 的情况，方差是无限的，需要特别处理，但通常为了有意义的方差，nu > 2。
    # 但根据要求，nu=1 也要生成，我们仍对其进行“标准化”操作，尽管其统计意义可能受限。
    raw_t_data = t.rvs(df=nu, size=(n_fixed_nu_vary, 1))

    # 标准化：零均值，单位方差
    # 对于nu=1（柯西分布）或nu=2，std会给出nan或inf，这里我们仍然执行计算，
    # 实际应用中可能需要更鲁棒的归一化方法，例如中位数和IQR
    if np.std(raw_t_data) > 1e-9: # 避免除以过小的标准差
        q_student_t_scaled = (raw_t_data - np.mean(raw_t_data)) / np.std(raw_t_data)
    else:
        q_student_t_scaled = raw_t_data - np.mean(raw_t_data) # 如果方差接近0，只去均值

    q_student_t_filename = os.path.join(output_dir, f'gaussian_vs_t_q_student_t_nu{nu}_n{n_fixed_nu_vary}.npy')
    np.save(q_student_t_filename, q_student_t_scaled)
    print(f"  已生成学生t分布样本 (nu={nu}): {q_student_t_filename}")

print("\n--- 情景二：固定自由度 nu=3，变化样本数量 n ---")
# --- 情景二：固定自由度 nu=3，变化样本数量 n ---
nu_fixed_n_vary = 3
sample_sizes = [25, 50, 100, 500, 1000, 5000, 10000]

for n_samples in sample_sizes:
    # 生成高斯分布数据 P
    p_gaussian = np.random.randn(n_samples, 1)
    p_gaussian_filename = os.path.join(output_dir, f'gaussian_vs_t_p_normal_n{n_samples}.npy')
    np.save(p_gaussian_filename, p_gaussian)

    # 生成学生 t 分布数据 Q，并进行标准化（零均值，单位方差）
    raw_t_data = t.rvs(df=nu_fixed_n_vary, size=(n_samples, 1))
    if np.std(raw_t_data) > 1e-9:
        q_student_t_scaled = (raw_t_data - np.mean(raw_t_data)) / np.std(raw_t_data)
    else:
        q_student_t_scaled = raw_t_data - np.mean(raw_t_data)

    q_student_t_filename = os.path.join(output_dir, f'gaussian_vs_t_q_student_t_nu{nu_fixed_n_vary}_n{n_samples}.npy')
    np.save(q_student_t_filename, q_student_t_scaled)

    print(f"  已生成 n={n_samples} 样本 (nu={nu_fixed_n_vary}): {p_gaussian_filename} 和 {q_student_t_filename}")

print("\n所有高斯分布与学生 t 分布区分实验数据生成完毕。")