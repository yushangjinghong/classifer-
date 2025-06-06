import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
from scipy.stats import norm
from sklearn.model_selection import train_test_split # For metal.train_test_split equivalent

# --- metal library equivalent in Python ---
class Metal:
    def get_rows(self, x, idx):
        if isinstance(x, (list, tuple)):
            res = []
            for item in x:
                res.append(self.get_rows(item, idx))
            return res
        # Adjusting idx for 0-based indexing from Lua's 1-based
        return x[idx - 1]

    def random_batches(self, x, y, bs=1):
        if isinstance(x, (list, tuple)):
            n = x[0].size(0)
        else:
            n = x.size(0)

        p = torch.randperm(n)
        index = 0
        
        while index < n:
            from_idx = index
            to_idx = min(index + bs - 1, n - 1)
            index += bs

            if to_idx < n:
                # Need to convert p indices to 0-based for slicing
                batch_x = self.get_rows(x, p[from_idx : to_idx + 1] + 1) # +1 because get_rows expects 1-based
                batch_y = self.get_rows(y, p[from_idx : to_idx + 1] + 1)
                yield batch_x, batch_y
            else:
                break # Ensure we don't yield partial batches if the loop logic doesn't handle it naturally

    def normalize(self, x, eps=0):
        y = x.clone()
        m = x.mean(dim=0, keepdim=True) # Lua's mean(1) is PyTorch's dim=0
        s = x.std(dim=0, keepdim=True) + eps
        y = (y - m) / s
        return y

    def train_test_split(self, x, y, p_tr=0.5):
        # Using scikit-learn's train_test_split for robust splitting
        # Ensure x and y are on CPU and are numpy arrays for sklearn
        x_np = x.cpu().numpy() if isinstance(x, torch.Tensor) else x
        y_np = y.cpu().numpy() if isinstance(y, torch.Tensor) else y

        x_tr, x_te, y_tr, y_te = train_test_split(x_np, y_np, train_size=p_tr, random_state=None)
        
        # Convert back to PyTorch tensors
        return (
            torch.from_numpy(x_tr).float(),
            torch.from_numpy(y_tr).float(),
            torch.from_numpy(x_te).float(),
            torch.from_numpy(y_te).float()
        )

    def train(self, net, criterion, x, y, parameters=None):
        parameters = parameters or {}
        gpu = parameters.get('gpu', False)
        verbose = parameters.get('verbose', False)
        batch_size = parameters.get('batchSize', 64)
        optimizer_class = parameters.get('optimizer', optim.SGD)
        
        device = torch.device('cuda' if gpu and torch.cuda.is_available() else 'cpu')
        net.to(device)
        criterion.to(device)

        # Initialize optimizer if not already done.
        # Lua's `optimState` handling implies a persistent optimizer state,
        # so we pass the optimizer instance around instead of re-creating if possible.
        # Here we'll create it for simplicity, assuming `optimState` is reset per call.
        if 'optimizer_instance' not in parameters:
             parameters['optimizer_instance'] = optimizer_class(net.parameters())
        optimizer = parameters['optimizer_instance']
        
        net.train()
        
        # Adjust y for NLLLoss/CrossEntropyLoss if needed (Lua's `nn.ClassNLLCriterion` maps here)
        if isinstance(criterion, (nn.NLLLoss, nn.CrossEntropyLoss)):
            # If target is float and criterion expects long, convert
            if y.dtype == torch.float:
                y = y.view(-1).long() # Ensure it's 1D and long type

        # Iterate over minibatches
        for batch_x, batch_y in self.random_batches(x, y, batch_size):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            # If BCELoss and target is (N,1), flatten to (N) if model output is (N)
            if isinstance(criterion, nn.BCELoss) and batch_y.dim() > 1 and batch_y.size(1) == 1:
                batch_y = batch_y.squeeze(1)
            
            optimizer.zero_grad()
            
            prediction = net(batch_x)
            if prediction.dim() > 1 and prediction.size(1) == 1:
                prediction = prediction.squeeze(1)
            loss = criterion(prediction, batch_y)
            loss.backward()
            optimizer.step()

            if verbose:
                # Python doesn't have xlua.progress. You can use tqdm or print manually.
                print(f"Batch processed. Loss: {loss.item():.4f}")

    def predict(self, net, x, parameters=None):
        parameters = parameters or {}
        batch_size = parameters.get('batchSize', 16)
        gpu = parameters.get('gpu', False)
        
        device = torch.device('cuda' if gpu and torch.cuda.is_available() else 'cpu')
        net.to(device)
        net.eval()
        
        all_predictions = []
        
        n = x.size(0) if isinstance(x, torch.Tensor) else x[0].size(0)
        
        with torch.no_grad():
            for i in range(0, n, batch_size):
                to_idx = min(i + batch_size, n)
                # Select rows for current batch. +1 for get_rows' 1-based expectation.
                batch_x = self.get_rows(x, torch.arange(i, to_idx) + 1).to(device)
                
                prediction = net(batch_x)
                
                # If output is single-dimensional (e.g., for BCE), ensure it's not (N,1)
                # and matches the expected output shape for concatenation
                if prediction.dim() == 1 and batch_size > 1: # If it's a 1D tensor for multiple items, make it (batch_size, 1)
                     prediction = prediction.unsqueeze(1)
                
                all_predictions.append(prediction.cpu()) # Move to CPU to accumulate

        if not all_predictions: # Handle case of empty input
            return torch.tensor([])

        # Concatenate predictions from all batches
        if isinstance(all_predictions[0], (list, tuple)): # For multiple outputs (not typical for this network)
            # This case is less common for simple sequential nets,
            # but would require zipping and concatenating each output.
            # Assuming a single tensor output for this specific neural_test setup.
            return torch.cat([p[0] for p in all_predictions], dim=0)
        else:
            return torch.cat(all_predictions, dim=0)

    def eval(self, net, criterion, x, y, parameters=None):
        parameters = parameters or {}
        gpu = parameters.get('gpu', False)

        device = torch.device('cuda' if gpu and torch.cuda.is_available() else 'cpu')
        net.to(device)
        criterion.to(device)
        net.eval() # Set to evaluation mode

        predictions = self.predict(net, x, parameters)
        
        # Ensure y is on CPU and correct type for criterion
        y_cpu = y.cpu()
        if isinstance(criterion, (nn.NLLLoss, nn.CrossEntropyLoss)):
            if y_cpu.dtype == torch.float:
                y_cpu = y_cpu.view(-1).long() # Ensure it's 1D and long type
        elif isinstance(criterion, nn.BCELoss):
             # BCE expects target shape to match output shape, often (N,) if output is (N,)
            if y_cpu.dim() > 1 and y_cpu.size(1) == 1:
                y_cpu = y_cpu.squeeze(1)
                
        if predictions.dim() > 1 and predictions.size(1) == 1:
            predictions = predictions.squeeze(1)

        loss = criterion(predictions.to(device), y_cpu.to(device)).item() # Calculate loss on device, then move to CPU for item()

        accuracy = None
        if isinstance(criterion, nn.BCELoss):
            # For BCE, predictions are usually probabilities (0-1), threshold at 0.5
            # Lua: torch.ge(predictions,0.5):long(), torch.eq(plabels,y:long()):double():mean()
            predicted_labels = (predictions >= 0.5).long()
            accuracy = (predicted_labels == y_cpu.long()).double().mean().item()
        elif isinstance(criterion, (nn.NLLLoss, nn.CrossEntropyLoss)):
            # For NLL/CrossEntropy, predictions are logits or log-probabilities.
            # Max along dim 1 gives the predicted class.
            # Lua: _, plabels = torch.max(predictions,2), torch.eq(plabels,y:view(-1):long()):double():mean()
            _, predicted_labels = torch.max(predictions, dim=1)
            accuracy = (predicted_labels == y_cpu.view(-1).long()).double().mean().item()
        
        return loss, accuracy

    def save(self, net, fname):
        net.eval()
        # In PyTorch, state_dict is typically saved
        torch.save(net.state_dict(), fname)

    def load(self, fname):
        # This function loads a pre-trained state_dict into a network.
        # The network architecture must be defined first.
        # This implementation simply loads the state_dict, but for actual use,
        # you'd need to create the model architecture and then load the state_dict into it.
        return torch.load(fname)

# Instantiate the Metal class
metal = Metal()

# --- Original Lua functions translated to Python ---

def make_dataset(p, q):
    # Lua: local x = torch.cat(p,q,1) -> Python: torch.cat([p, q], dim=0)
    x = torch.cat([p, q], dim=0)
    # Lua: local y = torch.cat(torch.zeros(p:size(1),1), torch.ones(q:size(1),1), 1)
    # PyTorch: zeros/ones have dim=0 for rows
    y = torch.cat([torch.zeros(p.size(0), 1), torch.ones(q.size(0), 1)], dim=0)
    
    # Lua's train_test_split is in metal, so we call it
    return metal.train_test_split(x, y)

def neural_test(p, q, epochs=100, h=20):
    x_tr, y_tr, x_te, y_te = make_dataset(p, q)

    # Lua: net:add(nn.Linear(x_tr:size(2),h)) -> PyTorch: x_tr.size(1) for features
    net = nn.Sequential(
        nn.Linear(x_tr.size(1), h),
        nn.ReLU(),
        nn.Linear(h, 1),
        nn.Sigmoid()
    )

    criterion = nn.BCELoss() # Lua: nn.BCECriterion() maps to BCELoss (assuming sigmoid output)

    # Lua: local params = { optimizer = optim.adam } -> Python: pass the class
    params = {'optimizer': optim.Adam}

    # In PyTorch, the optimizer instance needs to be created with net.parameters()
    # It will be created inside metal.train
    params['optimizer_instance'] = params['optimizer'](net.parameters())

    for i in range(epochs):
        metal.train(net, criterion, x_tr, y_tr, params)

    loss, acc = metal.eval(net, criterion, x_te, y_te, params)

    # Lua: local cdf = distributions.norm.cdf(acc,0.5,torch.sqrt(0.25/x_te:size(1)))
    # PyTorch: x_te.size(0) for num_samples
    # Using scipy.stats.norm.cdf as PyTorch doesn't have a direct equivalent
    
    # Ensure acc is a scalar for norm.cdf
    if isinstance(acc, torch.Tensor):
        acc = acc.item()

    std_dev = np.sqrt(0.25 / x_te.size(0))
    cdf = norm.cdf(acc, loc=0.5, scale=std_dev)
    
    return acc, 1.0 - cdf

def experiment(dir_path, name, seed):
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # --- START MODIFICATION HERE ---
    # 旧代码: file_path = f"{dir_path}{name}.t7"
    # 新代码: 加载 .pt 文件
    file_path = f"{dir_path}{name}.pt" # 更改文件扩展名为 .pt
    
    # 加载 .pt 文件。它可能是一个字典 {'p': tensor_p, 'q': tensor_q}
    # 或者直接是一个包含 tensor_p, tensor_q 的列表/元组
    d = torch.load(file_path,weights_only=True)

    # 根据您之前生成 .pt 文件的结构来提取 p 和 q
    # 如果您保存的是 {'p': p_tensor, 'q': q_tensor}，那么这样提取：
    p = d['p']
    q = d['q']
    
    # 如果您保存的是一个直接的张量对，例如 torch.save((p_tensor, q_tensor), output_file_name)
    # 那么您可能需要这样提取：
    # p = d[0]
    # q = d[1]
    # 请根据您实际的 .pt 文件内容调整
    # --- END MODIFICATION HERE ---

    # Ensure tensors are float32 for neural network operations
    p = p.float()
    q = q.float()

    if name == 'faces_same':
        i = torch.randperm(p.size(0)) # 0-indexed perm
        p = p[i] # Directly index with permuted indices
        
        # Split p into two halves. Ensure integer division.
        half_size = p.size(0) // 2
        q = p[half_size:]
        p = p[:half_size]
    
    # Balance dataset sizes
    if p.size(0) > q.size(0):
        i = torch.randperm(p.size(0))[:q.size(0)]
        p = p[i]
    elif q.size(0) > p.size(0):
        i = torch.randperm(q.size(0))[:p.size(0)]
        q = q[i]

    accuracy, p_value = neural_test(p, q)
    print(f"{seed}\t{name}\t{q.size(0)}\t{p.size(0)}\t{accuracy:.4f}\t{p_value:.4f}")

# --- Main execution block ---
if __name__ == '__main__':
    # dir_path 应该指向您存放 .pt 文件的目录
    # 根据您之前的路径，可能是：
    dir_path = '/root/autodl-tmp/classifer_l/' # 如果 .pt 文件直接在脚本同级目录
    # 或者如果它们在 /home/yushangjinghong/Desktop/classifer_l/data/ 下：
    # dir_path = '/home/yushangjinghong/Desktop/classifer_l/data/'

    # --- START MODIFICATION HERE ---
    # 调整 files 列表以匹配您生成的 .pt 文件名（不带 .pt 后缀）
    files = [
        'bayes_bayes',
        'bayes_deep',
        'bayes_learning',
        'bayes_neuro',
        'deep_neuro',
        'deep_learning',
        'neuro_learning',
        'faces_diff',
        'faces_same'
    ]
    # --- END MODIFICATION HERE ---

    parser = argparse.ArgumentParser(description='Run neural test experiments.')
    parser.add_argument('--seed', type=int, default=1, help='Random seed for reproducibility.')
    
    args = parser.parse_args()

    print("Seed\tName\tQ_Size\tP_Size\tAccuracy\tP_Value") # Header for output
    for f_name in files:
        experiment(dir_path, f_name, args.seed)