import torch
import math
from scipy.stats import norm # For cdf

def distances(x, y):
    """
    Computes the squared Euclidean distances between all pairs of rows in x and y.
    Equivalent to Lua's `distances` function.
    """
    result = x.new_full((x.size(0), y.size(0)), 0)
    result.addmm_(x, y.t(), beta=1, alpha=-2)
    result.add_(x.pow(2).sum(dim=1, keepdim=True))
    result.add_(y.pow(2).sum(dim=1, keepdim=True).t())
    return result

def make_dataset_split(p, q, p_tr=0.5):
    """
    Combines two datasets p and q, shuffles them, and splits into train/test sets.
    Labels are 0 for p and 1 for q.
    Equivalent to Lua's `make_dataset` that returns train/test splits.
    """
    x = torch.cat((p, q), dim=0)
    y = torch.cat((torch.zeros(p.size(0), 1), torch.ones(q.size(0), 1)), dim=0)
    
    n = x.size(0)
    perm = torch.randperm(n).long()
    
    x = x.index_select(0, perm).double() # Ensure double precision as in Lua
    y = y.index_select(0, perm).double()

    n_tr = math.floor(n * p_tr)
    
    x_tr = x[:n_tr].clone()
    y_tr = y[:n_tr].clone()
    x_te = x[n_tr:].clone()
    y_te = y[n_tr:].clone()

    return x_tr, y_tr, x_te, y_te

def knn_test(p, q, params=None):
    """
    Performs the KNN test for distinguishing between two datasets.
    Returns accuracy and p-value.
    `p` and `q` are assumed to be feature tensors.
    """
    params = params or {'ptr': 0.5} # ptr is proportion of training data
    x_tr, y_tr, x_te, y_te = make_dataset_split(p, q, params['ptr'])
    
    # Ensure inputs are float/double for distance calculation
    x_tr = x_tr.float() if x_tr.dtype == torch.float64 else x_tr
    x_te = x_te.float() if x_te.dtype == torch.float64 else x_te

    p_te_predictions = torch.zeros(x_te.size(0), dtype=torch.float)
    
    # k calculation as per Lua: math.sqrt(x_tr:size(1)) - should be x_tr.size(0) (number of samples)
    # The Lua code uses `x_tr:size(1)` which is the feature dimension, not num samples.
    # This seems like a potential bug or unconventional choice in the original Lua code.
    # Assuming it meant number of training samples for k.
    k_val = int(math.sqrt(x_tr.size(0)))
    if k_val == 0: k_val = 1 # Ensure k is at least 1
    
    # t is the threshold for majority vote: math.ceil(k/2)
    t_threshold = math.ceil(k_val / 2.0)
    
    # Calculate distances from test samples to training samples
    dist_matrix = distances(x_te, x_tr) # (num_test, num_train)

    # Get k nearest neighbors
    # sort_a: sorted distances, sort_b: indices of sorted distances
    _, sorted_indices = torch.sort(dist_matrix, dim=1, descending=False)
    
    # Select the indices of the k nearest neighbors for each test sample
    knn_indices = sorted_indices[:, :k_val] # (num_test, k)

    # For each test sample, check the labels of its k-nearest neighbors
    for i in range(x_te.size(0)):
        # Get labels of k-nearest neighbors from training set
        neighbor_labels = y_tr.index_select(0, knn_indices[i].long()).view(-1)
        
        # Sum of labels (0s and 1s). If sum > t_threshold, predict 1 (from q)
        if neighbor_labels.sum() > t_threshold:
            p_te_predictions[i] = 1
        else:
            p_te_predictions[i] = 0

    # Calculate accuracy (t)
    accuracy = torch.eq(p_te_predictions, y_te.view(-1)).float().mean().item()
    
    # Calculate p-value
    # Assuming null hypothesis: accuracy is 0.5 (random guess)
    # Standard deviation for binomial proportion: sqrt(p*(1-p)/n) where p=0.5
    if x_te.size(0) == 0:
        p_value = 1.0 # No test samples
    else:
        std_dev = math.sqrt(0.25 / x_te.size(0))
        # P(Z > observed_accuracy) = 1 - CDF(observed_accuracy | mean=0.5, std=std_dev)
        p_value = 1.0 - norm.cdf(accuracy, loc=0.5, scale=std_dev)
    
    return accuracy, p_value

if __name__ == '__main__':
    # Example usage for knn_test
    torch.manual_seed(42)
    
    # Generate synthetic data
    # Class 0: mean 0, Class 1: mean 1
    p_data = torch.randn(100, 20) * 0.5 + 0.0 # From distribution P
    q_data = torch.randn(100, 20) * 0.5 + 1.0 # From distribution Q
    
    print("Testing KNN with distinguishable distributions:")
    acc_dist, p_val_dist = knn_test(p_data, q_data)
    print(f"KNN Accuracy: {acc_dist:.4f}, P-value: {p_val_dist:.4f}") # Should be high accuracy, low p-value
    
    print("\nTesting KNN with less distinguishable distributions:")
    p_data_close = torch.randn(100, 20) * 0.5 + 0.0
    q_data_close = torch.randn(100, 20) * 0.5 + 0.1
    acc_close, p_val_close = knn_test(p_data_close, q_data_close)
    print(f"KNN Accuracy: {acc_close:.4f}, P-value: {p_val_close:.4f}") # Accuracy closer to 0.5, higher p-value
    
    print("\nTesting KNN with identical distributions:")
    acc_ident, p_val_ident = knn_test(p_data, p_data.clone()) # Should be close to 0.5, high p-value
    print(f"KNN Accuracy: {acc_ident:.4f}, P-value: {p_val_ident:.4f}")