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

# 移除 classifier 命令行参数，因为它将同时运行两种分类器
parser = argparse.ArgumentParser(description='Classifier Two-Sample Test for Type-I Error Control')
parser.add_argument('--test', type=int, default=100, help='每个样本数量重复测试的次数 (根据原文设定为100)')
parser.add_argument('--k', type=int, default=5, help='KNN 的邻居数量')
parser.add_argument('--hidden_layer', type=int, default=20, help='MLP 的隐藏层单元数量')
parser.add_argument('--epochs', type=int, default=100, help='MLP 的训练 epoch 数量')
parser.add_argument('--alpha', type=float, default=0.05, help='显著性水平 α')
parser.add_argument('--data_dir', type=str, default='type_i_error_data',
                    help='存储 .npy 文件的目录路径')


def load_data(filepath):
    # 从 .npy 文件加载数据
    return np.load(filepath)


def knn_test(p, q, k, test_size):
    p_data = p
    q_data = q
    X = np.vstack([p_data, q_data])
    y = np.concatenate([np.zeros(p_data.shape[0]), np.ones(q_data.shape[0])])

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

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, shuffle=True, random_state=None)

    X_tr_tensor = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_tensor = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)
    X_te_tensor = torch.tensor(X_te, dtype=torch.float32)
    y_te_tensor = torch.tensor(y_te, dtype=torch.float32).unsqueeze(1)

    model = MLPClassifier(X_tr_tensor.shape[1], hidden_layer)
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


def run_single_classifier_test(classifier_name, p_data, q_data, args):
    """
    运行单一分类器的测试，返回该分类器在所有样本数量下的I型错误率列表。
    """
    current_type_i_error_rates = []

    # 进行多次重复测试 (根据原文设定为100次)
    type_i_errors = 0
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

        if p_val <= args.alpha:
            type_i_errors += 1

    avg_acc = np.mean(accuracies)
    avg_p_value = np.mean(p_values)
    type_i_error_rate = type_i_errors / args.test

    return type_i_error_rate, avg_acc, avg_p_value


def main():
    args = parser.parse_args()

    sample_sizes = [25, 50, 100, 500, 1000, 5000, 10000]

    # 存储两种分类器的 I 型错误率
    knn_type_i_error_rates = []
    neural_type_i_error_rates = []

    print(f"I 型错误控制评估实验开始，显著性水平 α = {args.alpha}\n")

    for n_samples in sample_sizes:
        print(f"--- 样本数量 n = {n_samples} ---")
        p_filepath = os.path.join(args.data_dir, f'type_i_error_p_normal_{n_samples}_samples.npy')
        q_filepath = os.path.join(args.data_dir, f'type_i_error_q_normal_{n_samples}_samples.npy')

        if not os.path.exists(p_filepath) or not os.path.exists(q_filepath):
            print(f"错误：找不到样本数量为 {n_samples} 的数据文件。请确保 '{args.data_dir}' 目录中存在以下文件：")
            print(f"  {p_filepath}")
            print(f"  {q_filepath}")
            print("请先运行数据生成脚本。")
            knn_type_i_error_rates.append(np.nan)
            neural_type_i_error_rates.append(np.nan)
            continue

        p_data = load_data(p_filepath)
        q_data = load_data(q_filepath)

        # 运行 KNN 测试
        print(f"  正在运行 KNN 分类器...")
        knn_rate, knn_acc, knn_p_val = run_single_classifier_test('KNN', p_data, q_data, args)
        knn_type_i_error_rates.append(knn_rate)
        print(f"  KNN 分类器:")
        print(f"    平均分类准确度: {knn_acc:.4f}")
        print(f"    平均 P 值: {knn_p_val:.4f}")
        print(f"    I 型错误率 (在 α={args.alpha} 下): {knn_rate:.4f} ({int(knn_rate * args.test)}/{args.test} 次)")

        # 运行 Neural 测试
        print(f"  正在运行 Neural 分类器...")
        neural_rate, neural_acc, neural_p_val = run_single_classifier_test('Neural', p_data, q_data, args)
        neural_type_i_error_rates.append(neural_rate)
        print(f"  Neural 分类器:")
        print(f"    平均分类准确度: {neural_acc:.4f}")
        print(f"    平均 P 值: {neural_p_val:.4f}")
        print(f"    I 型错误率 (在 α={args.alpha} 下): {neural_rate:.4f} ({int(neural_rate * args.test)}/{args.test} 次)\n")


    print("I 型错误控制评估实验完成。")

    # --- 绘图部分 ---
    plt.figure(figsize=(12, 7)) # 稍微增大图表大小以适应多条曲线和图例
    plt.plot(sample_sizes, knn_type_i_error_rates, marker='o', linestyle='-', color='blue', label=f'KNN 实际 I 型错误率')
    plt.plot(sample_sizes, neural_type_i_error_rates, marker='x', linestyle='--', color='green', label=f'神经网络实际 I 型错误率')
    plt.axhline(y=args.alpha, color='red', linestyle=':', label=f'理论 α = {args.alpha}') # 虚线变为点线以区分

    plt.xscale('log') # 设置横轴为对数刻度，因为样本数量跨度很大
    plt.xlabel('样本数量 (n)')
    plt.ylabel('犯第一类错误的概率')
    plt.title(f'分类器在不同样本数量下的 I 型错误控制 (α={args.alpha})')
    plt.grid(True, which="both", ls="--", c='0.7')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()