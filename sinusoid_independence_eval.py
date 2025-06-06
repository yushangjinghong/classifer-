import argparse
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import norm
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# --- 设置中文字体 ---
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS'] # SimHei 是黑体，Arial Unicode MS 是一个常见的中文字体
plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题
# --- 中文字体设置结束 ---

parser = argparse.ArgumentParser(description='Classifier Two-Sample Test for Sinusoid Independence (Type-II Error)')
parser.add_argument('--test', type=int, default=100, help='每个样本组合重复测试的次数')
parser.add_argument('--k', type=int, default=5, help='KNN 的邻居数量')
parser.add_argument('--hidden_layer', type=int, default=20, help='MLP 的隐藏层单元数量')
parser.add_argument('--epochs', type=int, default=100, help='MLP 的训练 epoch 数量')
parser.add_argument('--alpha', type=float, default=0.05, help='显著性水平 α')
parser.add_argument('--data_dir', type=str, default='sinusoid_independence_data',
                    help='存储正弦波独立性测试 .npy 文件的目录路径')


def load_data(filepath):
    return np.load(filepath)


def knn_test(p, q, k, test_size):
    p_data = p
    q_data = q
    X = np.vstack([p_data, q_data])
    y = np.concatenate([np.zeros(p_data.shape[0]), np.ones(q_data.shape[0])])

    # 检查数据是否有效，防止空训练集/测试集
    if X.shape[0] < 2:
        return 0.5, 1.0
    if X.shape[0] * test_size < 1 or X.shape[0] * (1 - test_size) < 1:
        return 0.5, 1.0

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, shuffle=True, random_state=None)

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_tr, y_tr)

    y_pred = knn.predict(X_te)
    accuracy = accuracy_score(y_te, y_pred)

    n_te = X_te.shape[0]
    if n_te == 0:
        p_value = 1.0
    else:
        std_err = np.sqrt(0.25 / n_te)
        z_score = (accuracy - 0.5) / std_err
        p_value = 2 * min(norm.cdf(z_score), 1 - norm.cdf(z_score))

    return accuracy, p_value


class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

def neural_test(p, q, hidden_layer, epochs, test_size):
    p_data = p
    q_data = q
    X = np.vstack([p_data, q_data])
    y = np.concatenate([np.zeros(p_data.shape[0]), np.ones(q_data.shape[0])])

    # 检查数据是否有效，防止空训练集/测试集
    if X.shape[0] < 2:
        return 0.5, 1.0
    if X.shape[0] * test_size < 1 or X.shape[0] * (1 - test_size) < 1:
        return 0.5, 1.0

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, shuffle=True, random_state=None)

    X_tr_tensor = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_tensor = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)
    X_te_tensor = torch.tensor(X_te, dtype=torch.float32)
    y_te_tensor = torch.tensor(y_te, dtype=torch.float32).unsqueeze(1)

    # 检查输入大小是否有效，避免模型初始化错误
    input_size = X_tr_tensor.shape[1] if X_tr_tensor.numel() > 0 else 1 # 如果训练集为空，设置一个默认维度
    model = MLPClassifier(input_size, hidden_layer)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练 MLP 分类器
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_tr_tensor)
        loss = criterion(outputs, y_tr_tensor)
        loss.backward()
        optimizer.step()

    # 在测试集上评估分类器
    model.eval()
    with torch.no_grad():
        outputs = model(X_te_tensor)
    predictions = (outputs > 0.5).float()
    accuracy = (predictions == y_te_tensor).float().mean().item()

    n_te = X_te.shape[0]
    if n_te == 0:
        p_value = 1.0
    else:
        std_err = np.sqrt(0.25 / n_te)
        z_score = (accuracy - 0.5) / std_err
        p_value = 2 * min(norm.cdf(z_score), 1 - norm.cdf(z_score))

    return accuracy, p_value


def run_experiment_and_collect_errors(classifier_name, test_count, k_neighbors, hidden_units, epochs, alpha,
                                      data_directory, p_filenames, q_filenames):
    """
    运行实验并收集指定分类器的第二类错误率。
    """
    type_ii_error_rates = []

    for i in range(len(p_filenames)):
        p_filepath = os.path.join(data_directory, p_filenames[i])
        q_filepath = os.path.join(data_directory, q_filenames[i])

        if not os.path.exists(p_filepath) or not os.path.exists(q_filepath):
            print(f"警告：找不到数据文件。跳过 {p_filenames[i]} 和 {q_filenames[i]}")
            type_ii_error_rates.append(np.nan) # 标记为 NaN 以便绘图时处理
            continue

        p_data = load_data(p_filepath)
        q_data = load_data(q_filepath)

        type_ii_errors = 0 # 记录 Type-II 错误次数

        # 进行多次重复测试
        for _ in range(test_count):
            if classifier_name == 'KNN':
                acc, p_val = knn_test(p_data, q_data, k_neighbors, test_size=0.5)
            elif classifier_name == 'Neural':
                acc, p_val = neural_test(p_data, q_data, hidden_units, epochs, test_size=0.5)
            else:
                raise ValueError(f"错误：无法识别的分类器类型 '{classifier_name}'。")

            # 在此实验中，P (dependent) 和 Q (independent) 确实来自不同分布。
            # 原假设 H0 是 P=Q (即 X 和 Y 独立)。
            # 如果 P 值 > alpha，表示我们未能拒绝 H0，这构成了一次 Type-II 错误。
            if p_val > alpha:
                type_ii_errors += 1

        type_ii_error_rate = type_ii_errors / test_count
        type_ii_error_rates.append(type_ii_error_rate)
        # print(f"  {classifier_name} (Type-II): 错误率={type_ii_error_rate:.4f}") # 详细打印可以在这里

    return type_ii_error_rates


def main():
    args = parser.parse_args()

    # 从数据生成脚本中获取定义
    # 情景一：固定 n=2000, delta=1，变化 gamma (噪声标准差)
    n_fixed_dg_vary_gamma = 2000
    delta_fixed_dg_vary_gamma = 1
    gamma_values = [0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0]

    p_filenames_scenario1 = [f'sin_wave_p_dependent_n{n_fixed_dg_vary_gamma}_d{delta_fixed_dg_vary_gamma}_g{gamma:.2f}.npy' for gamma in gamma_values]
    q_filenames_scenario1 = [f'sin_wave_q_independent_n{n_fixed_dg_vary_gamma}_d{delta_fixed_dg_vary_gamma}_g{gamma:.2f}.npy' for gamma in gamma_values]

    # 情景二：固定 n=2000, gamma=0.25，变化 delta (频率)
    n_fixed_dg_vary_delta = 2000
    gamma_fixed_dg_vary_delta = 0.25
    delta_values = [1, 2, 4, 6, 8, 10, 20]

    p_filenames_scenario2 = [f'sin_wave_p_dependent_n{n_fixed_dg_vary_delta}_d{delta}_g{gamma_fixed_dg_vary_delta:.2f}.npy' for delta in delta_values]
    q_filenames_scenario2 = [f'sin_wave_q_independent_n{n_fixed_dg_vary_delta}_d{delta}_g{gamma_fixed_dg_vary_delta:.2f}.npy' for delta in delta_values]

    # 情景三：固定 delta=1, gamma=0.25，变化 n (样本数量)
    delta_fixed_n_vary = 1
    gamma_fixed_n_vary = 0.25
    n_values = [25, 50, 100, 500, 1000, 2000, 5000]

    p_filenames_scenario3 = [f'sin_wave_p_dependent_n{n}_d{delta_fixed_n_vary}_g{gamma_fixed_n_vary:.2f}.npy' for n in n_values]
    q_filenames_scenario3 = [f'sin_wave_q_independent_n{n}_d{delta_fixed_n_vary}_g{gamma_fixed_n_vary:.2f}.npy' for n in n_values]


    print(f"--- 正弦波上独立性测试 (第二类错误) 评估开始，显著性水平 α = {args.alpha} ---\n")

    # --- 绘图情景一：固定 delta=1, gamma=0.25，变化 n (样本数量) ---
    print(f"\n--- 运行情景一：固定频率和噪声标准差 (delta={delta_fixed_n_vary}, gamma={gamma_fixed_n_vary:.2f})，变化样本数量 (n) ---")
    knn_type_ii_error_n_vary = run_experiment_and_collect_errors('KNN', args.test, args.k, args.hidden_layer, args.epochs, args.alpha,
                                                                 args.data_dir, p_filenames_scenario3, q_filenames_scenario3)
    neural_type_ii_error_n_vary = run_experiment_and_collect_errors('Neural', args.test, args.k, args.hidden_layer, args.epochs, args.alpha,
                                                                    args.data_dir, p_filenames_scenario3, q_filenames_scenario3)

    plt.figure(figsize=(12, 7))
    plt.plot(n_values, knn_type_ii_error_n_vary, marker='o', linestyle='-', color='blue', label='KNN')
    plt.plot(n_values, neural_type_ii_error_n_vary, marker='x', linestyle='--', color='green', label='神经网络')
    # plt.xscale('log') # 移除对数刻度
    plt.xlabel('样本数量 (n)')
    plt.ylabel('犯第二类错误的概率')
    plt.title(f'分类器在不同样本数量下区分依赖与独立 (δ={delta_fixed_n_vary}, γ={gamma_fixed_n_vary:.2f}, α={args.alpha})')
    plt.grid(True, which="both", ls="--", c='0.7')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- 绘图情景二：固定 n=2000, delta=1，变化 gamma (噪声标准差) ---
    print(f"\n--- 运行情景二：固定样本数和频率 (n={n_fixed_dg_vary_gamma}, delta={delta_fixed_dg_vary_gamma})，变化噪声标准差 (gamma) ---")
    knn_type_ii_error_gamma_vary = run_experiment_and_collect_errors('KNN', args.test, args.k, args.hidden_layer, args.epochs, args.alpha,
                                                                     args.data_dir, p_filenames_scenario1, q_filenames_scenario1)
    neural_type_ii_error_gamma_vary = run_experiment_and_collect_errors('Neural', args.test, args.k, args.hidden_layer, args.epochs, args.alpha,
                                                                        args.data_dir, p_filenames_scenario1, q_filenames_scenario1)

    plt.figure(figsize=(12, 7))
    plt.plot(gamma_values, knn_type_ii_error_gamma_vary, marker='o', linestyle='-', color='blue', label='KNN')
    plt.plot(gamma_values, neural_type_ii_error_gamma_vary, marker='x', linestyle='--', color='green', label='神经网络')
    # plt.xscale('log') # 移除对数刻度
    plt.xlabel('噪声标准差 (γ)')
    plt.ylabel('犯第二类错误的概率')
    plt.title(f'分类器在不同噪声标准差下区分依赖与独立 (n={n_fixed_dg_vary_gamma}, δ={delta_fixed_dg_vary_gamma}, α={args.alpha})')
    plt.grid(True, which="both", ls="--", c='0.7')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- 绘图情景三：固定 n=2000, gamma=0.25，变化 delta (频率) ---
    print(f"\n--- 运行情景三：固定样本数和噪声标准差 (n={n_fixed_dg_vary_delta}, gamma={gamma_fixed_dg_vary_delta:.2f})，变化频率 (delta) ---")
    knn_type_ii_error_delta_vary = run_experiment_and_collect_errors('KNN', args.test, args.k, args.hidden_layer, args.epochs, args.alpha,
                                                                     args.data_dir, p_filenames_scenario2, q_filenames_scenario2)
    neural_type_ii_error_delta_vary = run_experiment_and_collect_errors('Neural', args.test, args.k, args.hidden_layer, args.epochs, args.alpha,
                                                                        args.data_dir, p_filenames_scenario2, q_filenames_scenario2)

    plt.figure(figsize=(12, 7))
    plt.plot(delta_values, knn_type_ii_error_delta_vary, marker='o', linestyle='-', color='blue', label='KNN')
    plt.plot(delta_values, neural_type_ii_error_delta_vary, marker='x', linestyle='--', color='green', label='神经网络')
    # plt.xscale('log') # 移除对数刻度
    plt.xlabel('频率 (δ)')
    plt.ylabel('犯第二类错误的概率')
    plt.title(f'分类器在不同频率下区分依赖与独立 (n={n_fixed_dg_vary_delta}, γ={gamma_fixed_dg_vary_delta:.2f}, α={args.alpha})')
    plt.grid(True, which="both", ls="--", c='0.7')
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("\n所有正弦波上独立性测试评估和绘图完毕。")


if __name__ == "__main__":
    main()