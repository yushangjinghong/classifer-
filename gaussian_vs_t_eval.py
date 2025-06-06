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

parser = argparse.ArgumentParser(description='Classifier Two-Sample Test for Gaussian vs. Student-t Discrimination (Type-II Error)')
parser.add_argument('--test', type=int, default=100, help='每个样本数量重复测试的次数')
parser.add_argument('--k', type=int, default=5, help='KNN 的邻居数量')
parser.add_argument('--hidden_layer', type=int, default=20, help='MLP 的隐藏层单元数量')
parser.add_argument('--epochs', type=int, default=100, help='MLP 的训练 epoch 数量')
parser.add_argument('--alpha', type=float, default=0.05, help='显著性水平 α')
parser.add_argument('--data_dir', type=str, default='gaussian_vs_t_data',
                    help='存储高斯 vs t 分布 .npy 文件的目录路径')


def load_data(filepath):
    return np.load(filepath)


def knn_test(p, q, k, test_size):
    p_data = p
    q_data = q
    X = np.vstack([p_data, q_data])
    y = np.concatenate([np.zeros(p_data.shape[0]), np.ones(q_data.shape[0])])

    # 检查 X_tr, X_te, y_tr, y_te 是否非空
    if X.shape[0] < 2: # 至少需要2个样本才能分割
        return 0.5, 1.0 # 无法训练和测试，返回默认值

    # 确保 test_size 不会导致空训练集或测试集
    if X.shape[0] * test_size < 1 or X.shape[0] * (1 - test_size) < 1:
        return 0.5, 1.0 # 数据量过小，无法有效分割，返回默认值

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

    # 检查 X_tr, X_te, y_tr, y_te 是否非空
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
    if X_tr_tensor.shape[0] == 0:
        return 0.5, 1.0 # 训练集为空，无法训练

    # 确保输入维度与模型一致
    input_size = X_tr_tensor.shape[1] if X_tr_tensor.numel() > 0 else 1 # 如果训练集为空，设置一个默认维度
    model = MLPClassifier(input_size, hidden_layer)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_tr_tensor)
        loss = criterion(outputs, y_tr_tensor)
        loss.backward()
        optimizer.step()

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


def run_classifier_trials_type_ii(classifier_name, p_data, q_data, args):
    """
    运行单一分类器的测试并计算第二类错误率 (针对 H0: 两个样本来自相同分布)。
    """
    type_ii_errors = 0
    accuracies = []
    p_values = []

    for i in range(args.test):
        if classifier_name == 'KNN':
            acc, p_val = knn_test(p_data, q_data, args.k, test_size=0.5)
        elif classifier_name == 'Neural':
            acc, p_val = neural_test(p_data, q_data, args.hidden_layer, args.epochs, test_size=0.5)
        else:
            raise ValueError(f"未知分类器类型: {classifier_name}")

        accuracies.append(acc)
        p_values.append(p_val)

        # 第二类错误: 原假设为假 (实际来自不同分布)，但未能拒绝原假设 (p_val > alpha)
        if p_val > args.alpha:
            type_ii_errors += 1

    avg_acc = np.mean(accuracies)
    avg_p_value = np.mean(p_values)
    error_rate = type_ii_errors / args.test

    return error_rate, avg_acc, avg_p_value


def main():
    args = parser.parse_args()

    # 从数据生成脚本中获取定义
    n_fixed_nu_vary = 2000
    nu_values = [1, 3, 5, 10, 15, 20, 30] # 自由度 (nu) 列表
    nu_fixed_n_vary = 3
    sample_sizes = [25, 50, 100, 500, 1000, 5000, 10000]

    print(f"--- 高斯 vs 学生 t 分布区分实验 (第二类错误) 开始，显著性水平 α = {args.alpha} ---\n")

    # --- 绘图情景一：固定自由度 nu=3，变化样本数量 n ---
    knn_type_ii_error_rates_n_vary = []
    neural_type_ii_error_rates_n_vary = []

    print(f"--- 情景一：固定自由度 nu={nu_fixed_n_vary}，变化样本数量 n ---")
    for n_samples in sample_sizes:
        print(f"--- 样本数量 n = {n_samples} (自由度 nu={nu_fixed_n_vary}) ---")
        p_filepath = os.path.join(args.data_dir, f'gaussian_vs_t_p_normal_n{n_samples}.npy')
        q_filepath = os.path.join(args.data_dir, f'gaussian_vs_t_q_student_t_nu{nu_fixed_n_vary}_n{n_samples}.npy')

        if not os.path.exists(p_filepath) or not os.path.exists(q_filepath):
            print(f"错误：找不到样本数量为 {n_samples}、自由度为 {nu_fixed_n_vary} 的数据文件。请确保 '{args.data_dir}' 目录中存在。")
            knn_type_ii_error_rates_n_vary.append(np.nan)
            neural_type_ii_error_rates_n_vary.append(np.nan)
            continue

        p_data = load_data(p_filepath)
        q_data = load_data(q_filepath)

        # 运行 KNN 测试 (第二类错误)
        print(f"  正在运行 KNN 分类器...")
        knn_rate_ii, knn_acc_ii, knn_p_val_ii = run_classifier_trials_type_ii('KNN', p_data, q_data, args)
        knn_type_ii_error_rates_n_vary.append(knn_rate_ii)
        print(f"  KNN 分类器: 错误率={knn_rate_ii:.4f}, 平均准确度={knn_acc_ii:.4f}, 平均 P 值={knn_p_val_ii:.4f}")

        # 运行 Neural 测试 (第二类错误)
        print(f"  正在运行 Neural 分类器...")
        neural_rate_ii, neural_acc_ii, neural_p_val_ii = run_classifier_trials_type_ii('Neural', p_data, q_data, args)
        neural_type_ii_error_rates_n_vary.append(neural_rate_ii)
        print(f"  Neural 分类器: 错误率={neural_rate_ii:.4f}, 平均准确度={neural_acc_ii:.4f}, 平均 P 值={neural_p_val_ii:.4f}\n")

    print(f"情景一评估完成 (固定自由度 nu={nu_fixed_n_vary}，变化样本数量)。\n")

    # --- 绘图：第二类错误概率 vs 样本数量 (固定自由度) ---
    plt.figure(figsize=(12, 7))
    plt.plot(sample_sizes, knn_type_ii_error_rates_n_vary, marker='o', linestyle='-', color='blue', label=f'KNN')
    plt.plot(sample_sizes, neural_type_ii_error_rates_n_vary, marker='x', linestyle='--', color='green', label=f'神经网络')
    plt.xscale('log')
    plt.xlabel('样本数量 (n)')
    plt.ylabel('犯第二类错误的概率')
    plt.title(f'分类器在不同样本数量下区分高斯和学生 t 分布 (nu={nu_fixed_n_vary}, α={args.alpha})')
    plt.grid(True, which="both", ls="--", c='0.7')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- 绘图情景二：固定样本数量 n=2000，变化自由度 nu ---
    knn_type_ii_error_rates_nu_vary = []
    neural_type_ii_error_rates_nu_vary = []

    print(f"--- 情景二：固定样本数 n={n_fixed_nu_vary}，变化自由度 nu ---")
    p_gaussian_fixed_n_path = os.path.join(args.data_dir, f'gaussian_vs_t_p_normal_n{n_fixed_nu_vary}.npy')
    # 只需要加载一次高斯分布数据
    if not os.path.exists(p_gaussian_fixed_n_path):
        print(f"错误：找不到固定样本数量为 {n_fixed_nu_vary} 的高斯数据文件。请确保 '{args.data_dir}' 目录中存在。")
        # 直接跳过整个情景二，或者用 NaN 填充
        knn_type_ii_error_rates_nu_vary = [np.nan] * len(nu_values)
        neural_type_ii_error_rates_nu_vary = [np.nan] * len(nu_values)
    else:
        p_data_fixed_n = load_data(p_gaussian_fixed_n_path)
        for nu in nu_values:
            print(f"--- 自由度 nu = {nu} (样本数量 n={n_fixed_nu_vary}) ---")
            q_filepath = os.path.join(args.data_dir, f'gaussian_vs_t_q_student_t_nu{nu}_n{n_fixed_nu_vary}.npy')

            if not os.path.exists(q_filepath):
                print(f"错误：找不到自由度为 {nu}、样本数量为 {n_fixed_nu_vary} 的学生 t 数据文件。请确保 '{args.data_dir}' 目录中存在。")
                knn_type_ii_error_rates_nu_vary.append(np.nan)
                neural_type_ii_error_rates_nu_vary.append(np.nan)
                continue

            q_data = load_data(q_filepath)

            # 运行 KNN 测试 (第二类错误)
            print(f"  正在运行 KNN 分类器...")
            knn_rate_ii_nu, knn_acc_ii_nu, knn_p_val_ii_nu = run_classifier_trials_type_ii('KNN', p_data_fixed_n, q_data, args)
            knn_type_ii_error_rates_nu_vary.append(knn_rate_ii_nu)
            print(f"  KNN 分类器: 错误率={knn_rate_ii_nu:.4f}, 平均准确度={knn_acc_ii_nu:.4f}, 平均 P 值={knn_p_val_ii_nu:.4f}")

            # 运行 Neural 测试 (第二类错误)
            print(f"  正在运行 Neural 分类器...")
            neural_rate_ii_nu, neural_acc_ii_nu, neural_p_val_ii_nu = run_classifier_trials_type_ii('Neural', p_data_fixed_n, q_data, args)
            neural_type_ii_error_rates_nu_vary.append(neural_rate_ii_nu)
            print(f"  Neural 分类器: 错误率={neural_rate_ii_nu:.4f}, 平均准确度={neural_acc_ii_nu:.4f}, 平均 P 值={neural_p_val_ii_nu:.4f}\n")

    print(f"情景二评估完成 (固定样本数量 n={n_fixed_nu_vary}，变化自由度)。\n")

    # --- 绘图：第二类错误概率 vs 自由度 (固定样本数量) ---
    plt.figure(figsize=(12, 7))
    plt.plot(nu_values, knn_type_ii_error_rates_nu_vary, marker='o', linestyle='-', color='blue', label=f'KNN')
    plt.plot(nu_values, neural_type_ii_error_rates_nu_vary, marker='x', linestyle='--', color='green', label=f'神经网络')
    # plt.xscale('log') # 自由度通常不是对数关系，如果范围跨度大可以考虑
    plt.xlabel('自由度 (nu)')
    plt.ylabel('犯第二类错误的概率')
    plt.title(f'分类器在不同自由度下区分高斯和学生 t 分布 (样本数 n={n_fixed_nu_vary}, α={args.alpha})')
    plt.grid(True, which="both", ls="--", c='0.7')
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("所有实验和绘图完成。")


if __name__ == "__main__":
    main()