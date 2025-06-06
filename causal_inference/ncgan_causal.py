import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import scale
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from tqdm import tqdm
import os
import math
import warnings
import pickle # Import pickle to handle .tkl/.pkl files

# Suppress warnings
warnings.filterwarnings("ignore")

# --- Command Line Argument Parsing (Same as before) ---
parser = argparse.ArgumentParser(description='Causal Inference using Conditional GANs with multiple test metrics')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--dz', type=int, default=10, help='dimension of latent noise')
parser.add_argument('--hiddens', type=int, default=128, help='number of hidden units')
parser.add_argument('--epochs', type=int, default=100, help='number of training epochs for GAN')
parser.add_argument('--bs', type=int, default=16, help='batch size')
parser.add_argument('--dr', type=float, default=0.25, help='dropout rate')
parser.add_argument('--plot', type=int, default=0, help='plot results (not implemented in this Python example)')
parser.add_argument('--tubingen_file', type=str, default='tubingen.tkl', help='path to the tubingen .tkl/.pkl file')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--beta1', type=float, default=0.5, help='Adam beta1 parameter')
parser.add_argument('--beta2', type=float, default=0.999, help='Adam beta2 parameter')
parser.add_argument('--wd', type=float, default=0, help='weight decay (L2 penalty)')
parser.add_argument('--remove_outliers', type=int, default=1, help='remove outliers')
parser.add_argument('--min', type=int, default=0, help='use min over reps (1) or average (0)')
parser.add_argument('--subsample', type=int, default=5000, help='subsample data points')
parser.add_argument('--reps', type=int, default=1, help='number of repetitions for one_cause_effect')
parser.add_argument('--ptr', type=float, default=0.5, help='proportion of data for test set/MMD kernel width etc.')
parser.add_argument('--test_epochs', type=int, default=10, help='epochs for neural test classifier')
parser.add_argument('--test_lr', type=float, default=0.001, help='learning rate for neural test classifier')

params = parser.parse_args()

torch.manual_seed(params.seed)
np.random.seed(params.seed)

# --- Data Loading and Preprocessing (Modified to handle .tkl) ---

def load_tuebingen_data(fname, params):
    """
    Loads the Tübigen dataset from a .tkl/.pkl file.
    Applies scaling and converts to PyTorch tensors.
    """
    try:
        with open(fname, 'rb') as f: # 'rb' for read binary mode
            data_x_list, data_y_labels, data_w_weights = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: {fname} not found. Please ensure the Tübigen data file exists.")
        return None, None, None
    except Exception as e:
        print(f"Error loading Tübigen data from {fname}: {e}")
        return None, None, None

    processed_pairs = []
    # original_y_labels = [] # Keep original labels if needed
    # original_w_weights = [] # Keep original weights if needed

    for i, pair_data in enumerate(data_x_list):
        # pair_data is expected to be a numpy array, typically N x 2
        # Apply outlier removal and subsampling similar to read_pair
        if params.remove_outliers == 1:
            pair_data = remove_outliers(pair_data)
            if pair_data.shape[0] == 0:
                print(f"Warning: All points removed as outliers from pair {i}. Skipping.")
                continue

        if params.subsample > 0 and pair_data.shape[0] > params.subsample:
            p = np.random.permutation(pair_data.shape[0])[:params.subsample]
            pair_data = pair_data[p]

        if pair_data.shape[1] != 2: # Ensure it has two columns (X and Y)
            print(f"Warning: Pair {i} in {fname} does not have 2 columns. Skipping.")
            continue

        # Normalize X and Y columns independently
        x_col = scale(pair_data[:, 0].reshape(-1, 1))
        y_col = scale(pair_data[:, 1].reshape(-1, 1))
        
        processed_pairs.append({
            'x': torch.tensor(x_col, dtype=torch.float32),
            'y': torch.tensor(y_col, dtype=torch.float32),
            'label': data_y_labels[i],
            'weight': data_w_weights[i]
        })
        # original_y_labels.append(data_y_labels[i])
        # original_w_weights.append(data_w_weights[i])

    print(f"Loaded {len(processed_pairs)} usable pairs from {fname}")
    return processed_pairs # Return a list of dicts, each containing 'x', 'y', 'label', 'weight'

def remove_outliers(x, k=20, a=0.05):
    if x.shape[0] < k:
        return x
    perm = np.random.permutation(x.shape[0])[:k]
    xk = x[perm].copy()
    d = np.full(x.shape[0], 1e6)
    for i in range(x.shape[0]):
        for j in range(xk.shape[0]):
            dij = np.linalg.norm(x[i] - xk[j])
            if dij < d[i]:
                d[i] = dij
    top = int(x.shape[0] * a)
    d_idx = np.argsort(d)[:-top]
    return x[d_idx].copy()


# --- Conditional GAN Model (Same as before) ---
class Generator(nn.Module):
    def __init__(self, Dx, Dy, dz, hiddens, dr):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(Dx + dz, hiddens),
            nn.ReLU(),
            nn.Dropout(dr),
            nn.Linear(hiddens, Dy)
        )
    def forward(self, x_z):
        return self.main(x_z)

class Discriminator(nn.Module):
    def __init__(self, Dx, Dy, hiddens, dr):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(Dx + Dy, hiddens),
            nn.ReLU(),
            nn.Dropout(dr),
            nn.Linear(hiddens, 1),
            nn.Sigmoid()
        )
    def forward(self, x_y):
        return self.main(x_y)

def train_conditional_gan(x, y, params):
    Dx = x.size(1)
    Dy = y.size(1)

    generator = Generator(Dx, Dy, params.dz, params.hiddens, params.dr)
    discriminator = Discriminator(Dx, Dy, params.hiddens, params.dr)

    optimizer_G = optim.Adam(generator.parameters(), lr=params.lr, betas=(params.beta1, params.beta2), weight_decay=params.wd)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=params.lr, betas=(params.beta1, params.beta2), weight_decay=params.wd)

    criterion = nn.BCELoss()

    real_label = 0.8
    fake_label = 0.2
    
    n_updates = 0
    
    dataset = torch.utils.data.TensorDataset(x, y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=params.bs, shuffle=True)

    # Adding a check for sufficiently large dataset for training
    if len(dataset) < params.bs:
        # print("Warning: Dataset too small for batch size. Skipping GAN training for this pair.")
        # Return dummy tensors if GAN training is skipped due to insufficient data
        return torch.randn(x.size(0), Dy) 

    for epoch in tqdm(range(params.epochs), desc="GAN Training"):
        for i, (b_x, b_y) in enumerate(dataloader):
            current_batch_size = b_x.size(0)

            b_z = torch.randn(current_batch_size, params.dz)
            b_xz = torch.cat((b_x, b_z), 1)

            # --- Train Discriminator ---
            discriminator.zero_grad()
            b_real_data = torch.cat((b_x, b_y), 1)
            output_real = discriminator(b_real_data).view(-1)
            errD_real = criterion(output_real, torch.full((current_batch_size,), real_label))
            errD_real.backward()

            fake_y = generator(b_xz).detach()
            b_fake_data = torch.cat((b_x, fake_y), 1)
            output_fake = discriminator(b_fake_data).view(-1)
            errD_fake = criterion(output_fake, torch.full((current_batch_size,), fake_label))
            errD_fake.backward()
            
            optimizer_D.step()

            # --- Train Generator ---
            if (n_updates % 2) == 0:
                generator.zero_grad()
                fake_y_for_G = generator(b_xz)
                b_fake_data_for_G = torch.cat((b_x, fake_y_for_G), 1)
                output_G = discriminator(b_fake_data_for_G).view(-1)
                errG = criterion(output_G, torch.full((current_batch_size,), real_label))
                errG.backward()
                optimizer_G.step()
            
            n_updates += 1

    z = torch.randn(x.size(0), params.dz) # Ensure z has correct number of rows for x.size(0)
    return generator(torch.cat((x, z), 1)).detach()

# --- Plotting function (placeholder) ---
def plot_results(x, y, px, py, c_xy, c_yx):
    pass

# --- Test Functions Implementation (Same as before) ---
def knn_test(data_real, data_fake, params):
    n_real = data_real.shape[0]
    n_fake = data_fake.shape[0]
    if n_real == 0 or n_fake == 0:
        return 0.5

    X_combined = torch.cat((data_real, data_fake), dim=0).cpu().numpy()
    y_combined = np.array([0] * n_real + [1] * n_fake)

    from sklearn.model_selection import train_test_split
    if X_combined.shape[0] < 2:
        return 0.5
        
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y_combined, test_size=params.ptr, random_state=params.seed, stratify=y_combined
    )
    
    if len(X_test) == 0:
        return 0.5

    knn_classifier = KNeighborsClassifier(n_neighbors=1)
    knn_classifier.fit(X_train, y_train)
    accuracy = knn_classifier.score(X_test, y_test)
    
    return accuracy

class NeuralClassifier(nn.Module):
    def __init__(self, input_dim, hiddens, dr):
        super(NeuralClassifier, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_dim, hiddens),
            nn.ReLU(),
            nn.Dropout(dr),
            nn.Linear(hiddens, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.main(x)

def neural_test(data_real, data_fake, params):
    n_real = data_real.shape[0]
    n_fake = data_fake.shape[0]
    if n_real == 0 or n_fake == 0:
        return 0.5

    X_combined = torch.cat((data_real, data_fake), dim=0)
    y_combined = torch.tensor([0.0] * n_real + [1.0] * n_fake, dtype=torch.float32).view(-1, 1)

    from sklearn.model_selection import train_test_split
    X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
        X_combined.cpu().numpy(), y_combined.cpu().numpy(), test_size=params.ptr, random_state=params.seed, stratify=y_combined.cpu().numpy()
    )
    
    if len(X_test_np) == 0:
        return 0.5

    X_train = torch.tensor(X_train_np, dtype=torch.float32)
    X_test = torch.tensor(X_test_np, dtype=torch.float32)
    y_train = torch.tensor(y_train_np, dtype=torch.float32)
    y_test = torch.tensor(y_test_np, dtype=torch.float32)

    input_dim = X_combined.size(1)
    classifier = NeuralClassifier(input_dim, params.hiddens, params.dr)
    optimizer = optim.Adam(classifier.parameters(), lr=params.test_lr)
    criterion = nn.BCELoss()

    test_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=params.bs, shuffle=True)

    if len(test_dataset) < params.bs: # Check if training data for test classifier is too small
        # print("Warning: Test classifier training data too small. Returning default accuracy.")
        return 0.5

    for epoch in range(params.test_epochs):
        for X_batch, y_batch in test_dataloader:
            optimizer.zero_grad()
            outputs = classifier(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        outputs = classifier(X_test)
        predicted = (outputs > 0.5).float()
        accuracy = (predicted == y_test).float().mean().item()
    
    return accuracy

def rbf_kernel(x, y, sigma=1.0):
    sq_dist = torch.cdist(x, y, p=2).pow(2)
    return torch.exp(-sq_dist / (2 * sigma**2))

def mmd_test(data_real, data_fake, params):
    sigma = params.ptr if params.ptr > 0 else 1.0 
    
    n_real = data_real.shape[0]
    n_fake = data_fake.shape[0]
    if n_real == 0 or n_fake == 0:
        return 1.0

    min_samples = min(n_real, n_fake)
    if min_samples == 0:
        return 1.0
    
    data_real_sample = data_real[torch.randperm(n_real)[:min_samples]]
    data_fake_sample = data_fake[torch.randperm(n_fake)[:min_samples]]

    K_xx = rbf_kernel(data_real_sample, data_real_sample, sigma)
    K_yy = rbf_kernel(data_fake_sample, data_fake_sample, sigma)
    K_xy = rbf_kernel(data_real_sample, data_fake_sample, sigma)

    mmd_val = (K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()).item()
    mmd_val = max(0, mmd_val) 
    
    return mmd_val

# --- Main Cause-Effect Inference Logic ---
def one_cause_effect(x, y, params):
    # Check if x or y is empty due to outlier removal/subsampling
    if x.shape[0] == 0 or y.shape[0] == 0:
        # print("Warning: Empty data after preprocessing. Returning default scores.")
        # If data is empty, return neutral scores that don't imply a direction
        if params._current_my_test in [0, 1]: return 0.5, 0.5 # For accuracy-based tests
        else: return 1.0, 1.0 # For MMD (high distance means no good fit)

    py = train_conditional_gan(x, y, params)
    px = train_conditional_gan(y, x, params)

    # Ensure generated data has correct shape in case GAN training was skipped
    if py.shape[0] != x.shape[0] or px.shape[0] != y.shape[0]:
        # print("Warning: Generated data has incorrect shape. Returning default scores.")
        if params._current_my_test in [0, 1]: return 0.5, 0.5
        else: return 1.0, 1.0

    current_test_func = None
    if params._current_my_test == 0:
        current_test_func = knn_test
    elif params._current_my_test == 1:
        current_test_func = neural_test
    elif params._current_my_test == 2:
        current_test_func = mmd_test
    else:
        raise ValueError("Invalid test type specified.")

    # Pass original and generated data to the test function
    c_xy_score_raw = current_test_func(torch.cat((x, y), dim=1), torch.cat((x, py), dim=1), params)
    c_yx_score_raw = current_test_func(torch.cat((x, y), dim=1), torch.cat((px, y), dim=1), params)

    if params._current_my_test in [0, 1]:
        c_xy = abs(c_xy_score_raw - 0.5)
        c_yx = abs(c_yx_score_raw - 0.5)
    elif params._current_my_test == 2:
        c_xy = c_xy_score_raw
        c_yx = c_yx_score_raw
        
    if params.plot == 1:
        plot_results(x, y, px, py, c_xy, c_yx)

    return c_xy, c_yx

def cause_effect(x, y, params):
    c_xy_scores = []
    c_yx_scores = []

    for i in range(params.reps):
        c_xy_i, c_yx_i = one_cause_effect(x, y, params)
        c_xy_scores.append(c_xy_i)
        c_yx_scores.append(c_yx_i)

    # Handle cases where all repetitions result in default scores (e.g., due to empty data)
    if not c_xy_scores: # If list is empty
        if params._current_my_test in [0, 1]: return 0.5, 0.5
        else: return 1.0, 1.0

    if params.min == 1:
        c_xy = min(c_xy_scores)
        c_yx = min(c_yx_scores)
    else:
        c_xy = sum(c_xy_scores) / len(c_xy_scores)
        c_yx = sum(c_yx_scores) / len(c_yx_scores)

    return c_xy, c_yx

# --- Main Execution Loop (Modified to load from .tkl) ---
def run_experiment(test_type, test_name, params):
    print(f"\n--- Running experiment with C2ST type: {test_name} ---")
    
    params._current_my_test = test_type 

    processed_tuebingen_pairs = load_tuebingen_data(params.tubingen_file, params)
    if processed_tuebingen_pairs is None or not processed_tuebingen_pairs:
        print("No usable Tübigen pairs loaded. Cannot run experiment.")
        return 0.0

    total_weight = 0.0
    correct_weighted_sum = 0.0
    
    # Iterate through the processed pairs from the .tkl file
    for i, pair_info in enumerate(tqdm(processed_tuebingen_pairs, desc=f"Processing Tübigen Pairs ({test_name})")):
        x = pair_info['x']
        y = pair_info['y']
        label = pair_info['label']
        weight = pair_info['weight']

        # Ensure the pair has enough samples for processing after subsampling/outlier removal
        if x.shape[0] < params.bs or y.shape[0] < params.bs: # Minimum for GAN training
             # print(f"Skipping pair {i} due to insufficient samples after preprocessing for GAN training.")
             continue # Skip if not enough samples

        c_xy, c_yx = cause_effect(x, y, params)
        
        # Prediction logic: Smaller score (c_xy or c_yx) indicates better fit/similarity,
        # implying that direction is more likely.
        predicted_label = 0 if c_xy < c_yx else 1
        
        result = (1 if predicted_label == label else 0) * weight
        
        total_weight += weight
        correct_weighted_sum += result
        
        # Optional: print individual results if desired
        # print(f'Pair {i} ({test_name}): Label={label}, Predicted={predicted_label}, Result={result:.5f}, c_xy={c_xy:.5f}, c_yx={c_yx:.5f}')

    final_accuracy = correct_weighted_sum / total_weight if total_weight > 0 else 0
    print(f"--- {test_name} Final Weighted Accuracy: {final_accuracy:.5f} ---")
    return final_accuracy

if __name__ == '__main__':
    # Create a dummy .tkl file if it doesn't exist for demonstration
    # This dummy will NOT produce meaningful results, but allows the code to run.
    # Replace this with your actual tubingen.tkl for real evaluation.
    if not os.path.exists(params.tubingen_file):
        print(f"Creating dummy {params.tubingen_file} for demonstration...")
        dummy_x_list = []
        dummy_y_labels = []
        dummy_w_weights = []
        
        np.random.seed(42)
        # Create a few dummy pairs
        # Pair 0: X->Y (label 0)
        x0 = np.random.randn(500, 1)
        y0 = 2*x0 + 0.1*np.random.randn(500,1) # Simple linear
        dummy_x_list.append(np.hstack((x0, y0)))
        dummy_y_labels.append(0)
        dummy_w_weights.append(1.0)

        # Pair 1: Y->X (label 1)
        y1 = np.random.randn(500, 1)
        x1 = np.sin(y1) + 0.2*np.random.randn(500,1) # Nonlinear from Y
        dummy_x_list.append(np.hstack((x1, y1)))
        dummy_y_labels.append(1)
        dummy_w_weights.append(1.0)

        # Pair 2: X->Y (label 0), another type
        x2 = np.random.rand(500,1) * 10
        y2 = np.exp(x2/5) + 0.5*np.random.randn(500,1)
        dummy_x_list.append(np.hstack((x2, y2)))
        dummy_y_labels.append(0)
        dummy_w_weights.append(0.8)

        # Pair 3: Small data, will be skipped by GAN training threshold
        x3 = np.random.randn(5,1)
        y3 = x3 + np.random.randn(5,1)
        dummy_x_list.append(np.hstack((x3, y3)))
        dummy_y_labels.append(0)
        dummy_w_weights.append(0.2)

        with open(params.tubingen_file, 'wb') as f:
            pickle.dump((dummy_x_list, dummy_y_labels, dummy_w_weights), f)
        print(f"Dummy {params.tubingen_file} created. /root/autodl-tmp/tuebingen.pkl")
    
    results = {}
    test_configs = {
        0: "KNN",
        1: "NN",
        2: "MMD"
    }

    # Run experiments for each test type
    for test_type, test_name in test_configs.items():
        # Reset seed for reproducibility for each run
        torch.manual_seed(params.seed)
        np.random.seed(params.seed)
        
        accuracy = run_experiment(test_type, test_name, params)
        results[test_name] = accuracy

    print("\n--- Summary of Results (CGAN-C2ST on Tübigen Pairs) ---")
    print("| C2ST type | Accuracy  |")
    print("|-----------|-----------|")
    for test_name, accuracy in results.items():
        print(f"| {test_name:<9} | {accuracy*100:.2f}%  |")
    print("|-----------|-----------|")

    print(f"\nNote: The accuracies will be meaningful only if you replace the dummy '{params.tubingen_file}' with the actual Tübigen dataset.")