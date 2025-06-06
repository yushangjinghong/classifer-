import numpy as np
import os

# 定义输出目录
output_dir = 'sinusoid_independence_data'
os.makedirs(output_dir, exist_ok=True)

print("开始生成正弦波上独立性测试数据...\n")

# --- 情景一：固定 n=2000, delta=1，变化 gamma (噪声标准差) ---
n_fixed_dg_vary_gamma = 2000
delta_fixed_dg_vary_gamma = 1
gamma_values = [0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0] # 噪声标准差列表

print(f"--- 情景一：固定 n={n_fixed_dg_vary_gamma}, delta={delta_fixed_dg_vary_gamma}，变化 gamma ---")
for gamma in gamma_values:
    x_data = np.random.randn(n_fixed_dg_vary_gamma, 1)
    epsilon_data = np.random.normal(loc=0, scale=gamma, size=(n_fixed_dg_vary_gamma, 1))
    y_data = np.cos(delta_fixed_dg_vary_gamma * x_data) + epsilon_data

    p_dependent = np.hstack([x_data, y_data])
    y_shuffled = y_data[np.random.permutation(n_fixed_dg_vary_gamma)]
    q_independent = np.hstack([x_data, y_shuffled])

    p_filename = os.path.join(output_dir, f'sin_wave_p_dependent_n{n_fixed_dg_vary_gamma}_d{delta_fixed_dg_vary_gamma}_g{gamma:.2f}.npy')
    q_filename = os.path.join(output_dir, f'sin_wave_q_independent_n{n_fixed_dg_vary_gamma}_d{delta_fixed_dg_vary_gamma}_g{gamma:.2f}.npy')

    np.save(p_filename, p_dependent)
    np.save(q_filename, q_independent)
    print(f"  已生成 gamma={gamma:.2f} 的数据: {p_filename} 和 {q_filename}")

# --- 情景二：固定 n=2000, gamma=0.25，变化 delta (频率) ---
n_fixed_dg_vary_delta = 2000
gamma_fixed_dg_vary_delta = 0.25
delta_values = [1, 2, 4, 6, 8, 10, 20] # 频率列表

print(f"\n--- 情景二：固定 n={n_fixed_dg_vary_delta}, gamma={gamma_fixed_dg_vary_delta}，变化 delta ---")
for delta in delta_values:
    x_data = np.random.randn(n_fixed_dg_vary_delta, 1)
    epsilon_data = np.random.normal(loc=0, scale=gamma_fixed_dg_vary_delta, size=(n_fixed_dg_vary_delta, 1))
    y_data = np.cos(delta * x_data) + epsilon_data

    p_dependent = np.hstack([x_data, y_data])
    y_shuffled = y_data[np.random.permutation(n_fixed_dg_vary_delta)]
    q_independent = np.hstack([x_data, y_shuffled])

    p_filename = os.path.join(output_dir, f'sin_wave_p_dependent_n{n_fixed_dg_vary_delta}_d{delta}_g{gamma_fixed_dg_vary_delta:.2f}.npy')
    q_filename = os.path.join(output_dir, f'sin_wave_q_independent_n{n_fixed_dg_vary_delta}_d{delta}_g{gamma_fixed_dg_vary_delta:.2f}.npy')

    np.save(p_filename, p_dependent)
    np.save(q_filename, q_independent)
    print(f"  已生成 delta={delta} 的数据: {p_filename} 和 {q_filename}")

# --- 情景三：固定 delta=1, gamma=0.25，变化 n (样本数量) ---
delta_fixed_n_vary = 1
gamma_fixed_n_vary = 0.25
n_values = [25, 50, 100, 500, 1000, 2000, 5000] # 样本数量列表

print(f"\n--- 情景三：固定 delta={delta_fixed_n_vary}, gamma={gamma_fixed_n_vary}，变化 n ---")
for n_samples in n_values:
    x_data = np.random.randn(n_samples, 1)
    epsilon_data = np.random.normal(loc=0, scale=gamma_fixed_n_vary, size=(n_samples, 1))
    y_data = np.cos(delta_fixed_n_vary * x_data) + epsilon_data

    p_dependent = np.hstack([x_data, y_data])
    y_shuffled = y_data[np.random.permutation(n_samples)]
    q_independent = np.hstack([x_data, y_shuffled])

    p_filename = os.path.join(output_dir, f'sin_wave_p_dependent_n{n_samples}_d{delta_fixed_n_vary}_g{gamma_fixed_n_vary:.2f}.npy')
    q_filename = os.path.join(output_dir, f'sin_wave_q_independent_n{n_samples}_d{delta_fixed_n_vary}_g{gamma_fixed_n_vary:.2f}.npy')

    np.save(p_filename, p_dependent)
    np.save(q_filename, q_independent)
    print(f"  已生成 n={n_samples} 的数据: {p_filename} 和 {q_filename}")

print("\n所有正弦波上独立性测试数据生成完毕。")