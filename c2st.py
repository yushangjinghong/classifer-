import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import norm
import torch

# 使用命令行参数来指定分类器类型、数据集路径、测试数量等参数
parser = argparse.ArgumentParser(description='Classifier Two-Sample Test')
parser.add_argument('--classifier', type=str, default='KNN', help='Classifier type (KNN or Neural)')
parser.add_argument('--data1', type=str, required=True, help='Path to first dataset file (.npy)')
parser.add_argument('--data2', type=str, required=True, help='Path to second dataset file (.npy)')
parser.add_argument('--test', type=int, default=10, help='Number of tests to perform')
parser.add_argument('--k', type=int, default=5, help='Number of neighbors for KNN')
parser.add_argument('--hidden_layer', type=int, default=20, help='Number of hidden units for MLP')
parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs for MLP')
parser.add_argument('--alpha', type=float, default=0.05, help='Significance level')


def load_data(filepath):
    # Load data from .npy file
    return np.load(filepath)


def knn_test(p, q, k, test_size):
    # Create a dataset by combining p and q
    p_data = p
    q_data = q
    X = np.vstack([p_data, q_data])
    y = np.concatenate([np.zeros(p_data.shape[0]), np.ones(q_data.shape[0])])

    # Split the dataset into training and testing sets
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, shuffle=True)

    # Train a KNN classifier
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_tr, y_tr)

    # Evaluate the classifier on the test set
    y_pred = knn.predict(X_te)
    accuracy = accuracy_score(y_te, y_pred)

    # Calculate the p-value
    n_te = X_te.shape[0]
    p_value = 1.0 - norm.cdf((accuracy - 0.5) / np.sqrt(0.25 / n_te))

    return accuracy, p_value


def neural_test(p, q, hidden_layer, epochs, test_size):
    # Create a dataset by combining p and q
    p_data = p
    q_data = q
    X = np.vstack([p_data, q_data])
    y = np.concatenate([np.zeros(p_data.shape[0]), np.ones(q_data.shape[0])])

    # Split the dataset into training and testing sets
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, shuffle=True)

    # Convert data to tensor format for PyTorch
    X_tr_tensor = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_tensor = torch.tensor(y_tr, dtype=torch.float32)
    X_te_tensor = torch.tensor(X_te, dtype=torch.float32)
    y_te_tensor = torch.tensor(y_te, dtype=torch.float32)

    # Define a simple MLP classifier using PyTorch
    class MLPClassifier(torch.nn.Module):
        def __init__(self, input_size, hidden_size):
            super(MLPClassifier, self).__init__()
            self.fc1 = torch.nn.Linear(input_size, hidden_size)
            self.fc2 = torch.nn.Linear(hidden_size, 1)
            self.sigmoid = torch.nn.Sigmoid()

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.sigmoid(self.fc2(x))
            return x

    # Initialize the model, loss function, and optimizer
    model = MLPClassifier(X_tr_tensor.shape[1], hidden_layer)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the MLP classifier
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_tr_tensor)
        loss = criterion(outputs.squeeze(), y_tr_tensor)
        loss.backward()
        optimizer.step()

    # Evaluate the classifier on the test set
    with torch.no_grad():
        outputs = model(X_te_tensor)
    predictions = (outputs.squeeze() > 0.5).float()
    accuracy = (predictions == y_te_tensor).float().mean().item()

    # Calculate the p-value
    n_te = X_te.shape[0]
    p_value = 1.0 - norm.cdf((accuracy - 0.5) / np.sqrt(0.25 / n_te))

    return accuracy, p_value


def main():
    args = parser.parse_args()

    # Load datasets
    p = load_data(args.data1)
    q = load_data(args.data2)

    # Determine the classifier type
    if args.classifier == 'KNN':
        acc, p_value = knn_test(p, q, args.k, 0.5)
    elif args.classifier == 'Neural':
        acc, p_value = neural_test(p, q, args.hidden_layer, args.epochs, 0.5)
    else:
        print(f"Classifier type '{args.classifier}' not recognized.")
        return

    # Print the results
    print(f"Classifier: {args.classifier}")
    print(f"Accuracy: {acc:.4f}")
    print(f"P-value: {p_value:.4f}")

    # Perform multiple tests if specified
    if args.test > 1:
        accs = []
        p_values = []
        for _ in range(args.test):
            if args.classifier == 'KNN':
                acc, p_value = knn_test(p, q, args.k, 0.5)
            elif args.classifier == 'Neural':
                acc, p_value = neural_test(p, q, args.hidden_layer, args.epochs, 0.5)
            accs.append(acc)
            p_values.append(p_value)

        avg_acc = np.mean(accs)
        avg_p_value = np.mean(p_values)
        print(f"\nAverage Results over {args.test} tests:")
        print(f"Average Accuracy: {avg_acc:.4f}")
        print(f"Average P-value: {avg_p_value:.4f}")


if __name__ == "__main__":
    main()

