import torch
import torch.nn as nn
import torch.optim as optim
import math
from scipy.stats import norm # For cdf

# Assuming a `metal` like functionality needs to be implemented or replaced.
# In PyTorch, we typically write the train/eval loops directly.

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

def neural_test(p, q, epochs=10000, h=20):
    """
    Performs a neural network classification test to distinguish between p and q.
    Returns accuracy and p-value.
    `p` and `q` are assumed to be feature tensors.
    """
    x_tr, y_tr, x_te, y_te = make_dataset_split(p, q)
    
    # Ensure tensors are float for PyTorch models and move to device
    x_tr = x_tr.float()
    y_tr = y_tr.float()
    x_te = x_te.float()
    y_te = y_te.float()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_tr = x_tr.to(device)
    y_tr = y_tr.to(device)
    x_te = x_te.to(device)
    y_te = y_te.to(device)

    net = nn.Sequential(
        nn.Linear(x_tr.size(1), h), # Input features to hidden layer
        nn.ReLU(),
        nn.Linear(h, 1), # Hidden to output (single logit)
        nn.Sigmoid() # Output probability
    ).to(device)

    criterion = nn.BCEWithLogitsLoss() # Combines Sigmoid and BCE. If Sigmoid is last layer, use BCELoss
                                        # Lua code uses `nn.Sigmoid()` and `nn.BCECriterion()`,
                                        # which implies BCELoss. Let's use BCELoss and ensure sigmoid output.
    # net = nn.Sequential(
    #     nn.Linear(x_tr.size(1), h),
    #     nn.ReLU(),
    #     nn.Linear(h, 1) # Output is raw logit
    # ).to(device)
    # criterion = nn.BCEWithLogitsLoss() # BCEWithLogitsLoss is numerically more stable

    # The Lua code uses Sigmoid then BCECriterion, so let's stick to that for direct translation
    # and use `nn.BCELoss`
    criterion = nn.BCELoss()


    optimizer = optim.Adam(net.parameters()) # Lua code uses optim.adam

    print(f"Training Neural Test Classifier for {epochs} epochs...")
    for i in range(1, epochs + 1):
        optimizer.zero_grad()
        outputs = net(x_tr)
        loss = criterion(outputs, y_tr)
        loss.backward()
        optimizer.step()
        # xlua.progress(i,epochs) equivalent:
        if i % (epochs // 10) == 0 or i == epochs:
            print(f"  Epoch {i}/{epochs}, Loss: {loss.item():.4f}")

    # Evaluation
    net.eval() # Set network to evaluation mode
    with torch.no_grad(): # Disable gradient calculation
        outputs_te = net(x_te)
        
        # Calculate loss (for completeness, though accuracy is main metric)
        loss_te = criterion(outputs_te, y_te).item()
        
        # Predictions (0 or 1) based on Sigmoid output
        predictions = (outputs_te > 0.5).float()
        
        # Accuracy
        correct = (predictions == y_te).sum().item()
        accuracy = correct / x_te.size(0) if x_te.size(0) > 0 else 0.0
    
    # Calculate p-value
    # Assuming null hypothesis: accuracy is 0.5 (random guess)
    if x_te.size(0) == 0:
        p_value = 1.0
    else:
        std_dev = math.sqrt(0.25 / x_te.size(0))
        p_value = 1.0 - norm.cdf(accuracy, loc=0.5, scale=std_dev)
    
    return accuracy, p_value

if __name__ == '__main__':
    # Example usage for neural_test
    torch.manual_seed(42)
    
    # Generate synthetic data
    p_data = torch.randn(500, 50) * 0.5 + 0.0 # From distribution P
    q_data = torch.randn(500, 50) * 0.5 + 1.0 # From distribution Q
    
    print("Testing Neural Test with distinguishable distributions:")
    acc_dist, p_val_dist = neural_test(p_data, q_data)
    print(f"Neural Test Accuracy: {acc_dist:.4f}, P-value: {p_val_dist:.4f}") # Should be high accuracy, low p-value
    
    print("\nTesting Neural Test with less distinguishable distributions:")
    p_data_close = torch.randn(500, 50) * 0.5 + 0.0
    q_data_close = torch.randn(500, 50) * 0.5 + 0.1
    acc_close, p_val_close = neural_test(p_data_close, q_data_close, epochs=5000) # Fewer epochs for subtle diff
    print(f"Neural Test Accuracy: {acc_close:.4f}, P-value: {p_val_close:.4f}") # Accuracy closer to 0.5, higher p-value
    
    print("\nTesting Neural Test with identical distributions:")
    acc_ident, p_val_ident = neural_test(p_data, p_data.clone(), epochs=1000) # Should be close to 0.5, high p-value
    print(f"Neural Test Accuracy: {acc_ident:.4f}, P-value: {p_val_ident:.4f}")