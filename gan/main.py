import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models, datasets
from torchvision.utils import save_image, make_grid
import os
import math
from collections import defaultdict
from scipy.stats import norm # For cdf

# --- Helper Functions for Statistical Tests ---

def distances(x, y):
    """
    Computes the squared Euclidean distances between all pairs of rows in x and y.
    x: (N, D), y: (M, D)
    result: (N, M)
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

# --- KNN Test ---
def knn_test(p, q, params=None):
    """
    Performs the KNN test for distinguishing between two datasets.
    Returns accuracy and p-value.
    `p` and `q` are assumed to be feature tensors.
    """
    params = params or {'ptr': 0.5} # ptr is proportion of training data
    x_tr, y_tr, x_te, y_te = make_dataset_split(p, q, params['ptr'])
    
    # Ensure inputs are float for distance calculation and GPU compatibility
    x_tr = x_tr.float().cuda() if torch.cuda.is_available() else x_tr.float()
    x_te = x_te.float().cuda() if torch.cuda.is_available() else x_te.float()
    y_tr = y_tr.float().cuda() if torch.cuda.is_available() else y_tr.float()
    y_te = y_te.float().cuda() if torch.cuda.is_available() else y_te.float()

    if x_te.size(0) == 0:
        return float('nan'), 1.0 # No test samples

    p_te_predictions = torch.zeros(x_te.size(0), dtype=torch.float, device=x_te.device)
    
    k_val = int(math.sqrt(x_tr.size(0)))
    if k_val == 0: k_val = 1 # Ensure k is at least 1
    
    t_threshold = math.ceil(k_val / 2.0)
    
    dist_matrix = distances(x_te, x_tr) # (num_test, num_train)

    _, sorted_indices = torch.sort(dist_matrix, dim=1, descending=False)
    
    knn_indices = sorted_indices[:, :k_val] # (num_test, k)

    for i in range(x_te.size(0)):
        neighbor_labels = y_tr.index_select(0, knn_indices[i].long()).view(-1)
        
        if neighbor_labels.sum() > t_threshold:
            p_te_predictions[i] = 1
        else:
            p_te_predictions[i] = 0

    accuracy = torch.eq(p_te_predictions, y_te.view(-1)).float().mean().item()
    
    std_dev = math.sqrt(0.25 / x_te.size(0))
    p_value = 1.0 - norm.cdf(accuracy, loc=0.5, scale=std_dev)
    
    return accuracy, p_value

# --- Neural Test ---
def neural_test(p, q, epochs=10000, h=20):
    """
    Performs a neural network classification test to distinguish between p and q.
    Returns accuracy and p-value.
    `p` and `q` are assumed to be feature tensors.
    """
    x_tr, y_tr, x_te, y_te = make_dataset_split(p, q)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_tr = x_tr.float().to(device)
    y_tr = y_tr.float().to(device)
    x_te = x_te.float().to(device)
    y_te = y_te.float().to(device)

    if x_te.size(0) == 0:
        return float('nan'), 1.0 # No test samples

    net = nn.Sequential(
        nn.Linear(x_tr.size(1), h), # Input features to hidden layer
        nn.ReLU(),
        nn.Linear(h, 1), # Hidden to output (single logit)
        nn.Sigmoid() # Output probability
    ).to(device)

    criterion = nn.BCELoss() # Using BCELoss as per Lua equivalent
    optimizer = optim.Adam(net.parameters())

    # print(f"  Training Neural Test Classifier for {epochs} epochs...")
    for i in range(1, epochs + 1):
        optimizer.zero_grad()
        outputs = net(x_tr)
        loss = criterion(outputs, y_tr)
        loss.backward()
        optimizer.step()

    # Evaluation
    net.eval() # Set network to evaluation mode
    with torch.no_grad(): # Disable gradient calculation
        outputs_te = net(x_te)
        predictions = (outputs_te > 0.5).float()
        
        correct = (predictions == y_te).sum().item()
        accuracy = correct / x_te.size(0)
    
    std_dev = math.sqrt(0.25 / x_te.size(0))
    p_value = 1.0 - norm.cdf(accuracy, loc=0.5, scale=std_dev)
    
    return accuracy, p_value


# --- MMD Test ---
# --- Helper Function for distances (ensure this is present too) ---
def distances(x, y):
    """
    Computes the squared Euclidean distances between all pairs of rows in x and y.
    x: (N, D), y: (M, D)
    result: (N, M)
    """
    result = x.new_full((x.size(0), y.size(0)), 0)
    result.addmm_(x, y.t(), beta=1, alpha=-2)
    result.add_(x.pow(2).sum(dim=1, keepdim=True))
    result.add_(y.pow(2).sum(dim=1, keepdim=True).t())
    return result

# --- MMD Test Helper Functions ---

def rbf_kernel(x, y, gamma):
    """
    Computes the RBF kernel matrix K(x,y) = exp(-gamma * ||x-y||^2).
    """
    if gamma <= 0:
        # Gamma should always be positive from choose_g's selection
        raise ValueError("Gamma for RBF kernel must be positive.")
    
    sq_dist = distances(x, y)
    sq_dist[sq_dist < 0] = 0 # Ensure non-negative due to float precision
    
    return torch.exp(-gamma * sq_dist)

def mmd_stat(x, y, g_std):
    """
    Calculates the MMD statistic and its variance using a U-statistic.
    `g_std` is the standard deviation parameter for the RBF kernel.
    """
    m = x.size(0) # Number of samples in the current subset (either full or half)
    if (m % 2) == 1:
        m -= 1 # ensure m is even
    
    # Crucial check: After ensuring m is even, it must be at least 2 to form valid pairs for U-statistic.
    if m < 2:
        # Not enough samples to form the U-statistic pairs (x1,x2,y1,y2 all non-empty).
        # Return a small non-zero variance to prevent division by zero / NaN in choose_g or p-value calc.
        return torch.tensor(0.0), torch.tensor(1e-9) 

    m2_int = int(m / 2) 

    # Calculate gamma from g_std for rbf_kernel
    gamma = 1.0 / (2 * (g_std**2))

    # Split data for U-statistic. These slices will now have size m2_int, which is >= 1.
    x1 = x[:m2_int]
    x2 = x[m2_int:m]
    y1 = y[:m2_int]
    y2 = y[m2_int:m]

    kxx = rbf_kernel(x1, x2, gamma)
    kyy = rbf_kernel(y1, y2, gamma)
    kxy = rbf_kernel(x1, y2, gamma)
    kyx = rbf_kernel(x2, y1, gamma) 

    res = kxx + kyy - kxy - kyx # This is where your previous error was occurring
    
    # Ensure inputs are float for mean/var
    res = res.float() if res.dtype == torch.float64 else res 
    
    mean = res.mean()
    variance = res.var() / m2_int # As per Lua code's `res:var()/m2`
    
    return mean, variance

def choose_g(x, y):
    """
    Heuristically chooses the best 'g' (std) parameter for the RBF kernel based on max (mean/sqrt(var)).
    """
    g_values = torch.pow(2, torch.arange(-15.0, 11.0)) # Powers of 2 from 2^-15 to 2^10
    epsilon = 1e-4 # Small constant for stability

    best_g = 1.0
    best_ratio = -float('inf') # Initialize with negative infinity

    # mmd_stat internally needs at least 2 samples from x and y to produce meaningful results.
    # If the input x, y here are too small, mmd_stat will return 0.0, 1e-9.
    # We should still iterate to find a 'g' even if ratio is initially bad.

    for g_val in g_values:
        mean_stat, variance_stat = mmd_stat(x, y, g_val) 
        
        # Avoid division by zero if variance is too small or zero
        if variance_stat <= 0: # This case now implies variance_stat is 1e-9 from mmd_stat's check
            ratio = mean_stat / (torch.sqrt(torch.tensor(1e-9)) + epsilon) # Use fixed small variance
        else:
            denominator = torch.sqrt(variance_stat) + epsilon 
            ratio = mean_stat / denominator
        
        if ratio > best_ratio:
            best_ratio = ratio
            best_g = g_val
            
    return best_g

def mmd_test(x, y):
    """
    Performs the MMD test.
    Returns the MMD statistic and the p-value.
    """
    # Ensure inputs are on CPU for numpy.cdf later
    x = x.cpu()
    y = y.cpu()

    m = x.size(0)
    if (m % 2) == 1:
        m -= 1 # Ensure m is even
    
    # If after making 'm' even, it's still less than 2, we can't perform the test meaningfully
    if m < 2: 
        return float('nan'), 1.0 # Not enough samples for a valid test (cannot split for U-statistic)

    m2_int = int(m / 2) # Overall m2 for splitting x, y into two halves

    # Choose 'g' using the first half of the data, as per Lua
    g = choose_g(x[:m2_int], y[:m2_int])

    # Calculate MMD statistic and variance using the second half of the data
    mmd_mean, mmd_variance = mmd_stat(x[m2_int:m], y[m2_int:m], g)
    
    # Calculate p-value.
    # Handle very small or zero variance to avoid `nan` or `inf`
    if mmd_variance <= 0: # This should now generally be true only if mmd_stat returned 1e-9
        p_value = 1.0 if mmd_mean.item() <= 0 else 0.0 
    else:
        std_dev = math.sqrt(mmd_variance.item()) # .item() to convert tensor to scalar
        p_value = 1.0 - norm.cdf(mmd_mean.item(), loc=0, scale=std_dev)
    
    return mmd_mean.item(), p_value
# --- Argument Parsing ---
def parse_args():
    parser = argparse.ArgumentParser(description='Image quality assessment for CIFAR-10 GAN.')
    parser.add_argument('--my_test', type=str, default='mmd', help='Type of test to run: mmd, knn, or neural')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--base_dir', type=str, default='.', help='Base directory for datasets and models')
    parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset name (only CIFAR-10 is assumed for now)')
    parser.add_argument('--features', type=str, default='resnet', help='Features to use: pixels or resnet')
    parser.add_argument('--collage', type=int, default=64, help='Number of images for collage')
    parser.add_argument('--gf', type=int, default=32, help='Generator filter size (for file naming)')
    parser.add_argument('--df', type=int, default=32, help='Discriminator filter size (for file naming)')
    parser.add_argument('--ep', type=int, default=199, help='Epochs (for file naming)')
    parser.add_argument('--max_n', type=int, default=5000, help='Maximum number of samples per class for evaluation (0 for all)')
    parser.add_argument('--bs', type=int, default=128, help='Batch size for featurization')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of classes in the dataset (CIFAR-10 is 10)')
    # Neural test specific arguments
    parser.add_argument('--neural_epochs', type=int, default=5000, help='Epochs for Neural Test classifier training')
    parser.add_argument('--neural_hidden', type=int, default=20, help='Hidden layer size for Neural Test classifier')
    return parser.parse_args()

# --- Feature Extraction ---
def featurize(net, imgs_tensor, batch_size):
    """
    Extracts features from images using a pre-trained network.
    """
    if imgs_tensor.size(0) == 0:
        return torch.empty(0, 2048) # Return empty tensor if no images

    preprocess = transforms.Compose([
        transforms.Resize(224), # ResNet input size
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    predictions = torch.zeros(imgs_tensor.size(0), 2048)
    net.eval()
    net.cuda()

    for n in range(0, imgs_tensor.size(0), batch_size):
        to = min(imgs_tensor.size(0), n + batch_size)
        batch_imgs = imgs_tensor[n:to].cuda()
        
        batch_imgs = preprocess(batch_imgs) 
        
        with torch.no_grad():
            output = net(batch_imgs)
        predictions[n:to] = output.float().cpu()
    return predictions

# --- Image Collage Saving ---
def save_image_collage(tensor, filename, nrow):
    """
    Saves a tensor of images as a collage to a file.
    Expects tensor to be in range [-1, 1] and normalizes to [0, 1] for saving.
    """
    if tensor.dim() == 4 and tensor.size(0) > 0:
        display_tensor = (tensor + 1) / 2 # Denormalize from [-1, 1] to [0, 1]
        grid = make_grid(display_tensor, nrow=nrow, padding=2)
        save_image(grid, filename)
    else:
        print(f"Warning: Cannot save collage. Expected 4D tensor (NCHW) with samples, got {tensor.dim()}D tensor with {tensor.size(0)} samples.")

# --- Main Execution ---
def main():
    params = parse_args()
    
    torch.manual_seed(params.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(params.seed)

    # --- Load Real Samples (and their labels from CIFAR-10 dataset directly) ---
    print("Loading real CIFAR-10 dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # Normalize to [-1, 1] for consistency with GAN output
    ])
    full_real_dataset = datasets.CIFAR10(
        root=os.path.join(params.base_dir, 'data'), 
        train=True, 
        download=True, 
        transform=transform
    )
    
    real_images_all = torch.stack([img for img, _ in full_real_dataset])
    real_labels_all = torch.tensor([label for _, label in full_real_dataset])
    print(f'Read {real_images_all.size(0)} real {params.dataset} samples with labels.')

    os.makedirs(os.path.join(params.base_dir, 'generated_images'), exist_ok=True)
    p_idx = torch.randperm(real_images_all.size(0))[:params.collage].long()
    save_image_collage(real_images_all.index_select(0, p_idx), 
                       os.path.join(params.base_dir, 'generated_images', 'real_cifar10_collage.png'), 
                       nrow=8)

    # --- Load Fake Samples (assuming labels are saved alongside images) ---
    fake_data_file = f'conditional_samples_{params.dataset}.pt' 
    fake_dir = os.path.join(params.base_dir, 'generated_conditional_images', fake_data_file)
    
    if not os.path.exists(fake_dir):
        raise FileNotFoundError(f"Fake dataset with labels not found at {fake_dir}. "
                                "Ensure your conditional GAN generation script saves labels alongside images "
                                "in a dictionary with 'images' and 'labels' keys.")
    
    try:
        fake_data = torch.load(fake_dir)
        fake_images_all = fake_data['images']
        fake_labels_all = fake_data['labels']
        print(f'Read {fake_images_all.size(0)} fake {params.dataset} samples with labels.')
    except Exception as e:
        raise Exception(f"Error loading fake samples from {fake_dir}: {e}. "
                        "Expected a dict with 'images' and 'labels' keys.")

    p_idx = torch.randperm(fake_images_all.size(0))[:params.collage].long()
    save_image_collage(fake_images_all.index_select(0, p_idx), 
                       os.path.join(params.base_dir, 'generated_images', 'fake_cifar10_conditional_collage.png'), 
                       nrow=8)

    # --- Prepare for Per-Class Comparison ---
    real_samples_by_class = defaultdict(list)
    fake_samples_by_class = defaultdict(list)

    for img, label in zip(real_images_all, real_labels_all):
        real_samples_by_class[label.item()].append(img)
    for k in real_samples_by_class.keys():
        if len(real_samples_by_class[k]) > 0:
            real_samples_by_class[k] = torch.stack(real_samples_by_class[k])
            if params.max_n > 0:
                n_real_class = min(real_samples_by_class[k].size(0), params.max_n)
                real_samples_by_class[k] = real_samples_by_class[k].index_select(
                    0, torch.randperm(real_samples_by_class[k].size(0))[:n_real_class].long()
                ).float()
        else:
            real_samples_by_class[k] = torch.empty(0, real_images_all.size(1), real_images_all.size(2), real_images_all.size(3))

    for img, label in zip(fake_images_all, fake_labels_all):
        fake_samples_by_class[label.item()].append(img)
    for k in fake_samples_by_class.keys():
        if len(fake_samples_by_class[k]) > 0:
            fake_samples_by_class[k] = torch.stack(fake_samples_by_class[k])
            if params.max_n > 0:
                n_fake_class = min(fake_samples_by_class[k].size(0), params.max_n)
                fake_samples_by_class[k] = fake_samples_by_class[k].index_select(
                    0, torch.randperm(fake_samples_by_class[k].size(0))[:n_fake_class].long()
                ).float()
        else:
            fake_samples_by_class[k] = torch.empty(0, fake_images_all.size(1), fake_images_all.size(2), fake_images_all.size(3))

    # --- Featurization (ResNet) or Pixel-wise (flatten) ---
    real_samples_processed = {}
    fake_samples_processed = {}

    if params.features == 'resnet':
        print(f"\nFeaturizing samples using ResNet50...")
        net = models.resnet50(pretrained=True)
        net = nn.Sequential(*list(net.children())[:-1])
        net.add_module('flatten', nn.Flatten())
        net.eval()
        net.cuda()
        
        for c in range(params.num_classes):
            print(f"  Featurizing real samples for class {c} (count: {real_samples_by_class[c].size(0)})")
            real_samples_processed[c] = featurize(net, real_samples_by_class[c], params.bs)
            print(f"  Featurizing fake samples for class {c} (count: {fake_samples_by_class[c].size(0)})")
            fake_samples_processed[c] = featurize(net, fake_samples_by_class[c], params.bs)
        
        torch.save({'real_features_by_class': real_samples_processed, 
                    'fake_features_by_class': fake_samples_processed},
                   os.path.join(params.base_dir, f'featurized_{params.dataset}_resnet.pt'))
    else: # Use raw pixels
        print("\nUsing raw pixel features (flattening images)...")
        for c in range(params.num_classes):
            real_samples_processed[c] = real_samples_by_class[c].view(real_samples_by_class[c].size(0), -1).clone()
            fake_samples_processed[c] = fake_samples_by_class[c].view(fake_samples_by_class[c].size(0), -1).clone()
        
        torch.save({'real_samples_processed': real_samples_processed, 
                    'fake_samples_processed': fake_samples_processed},
                   os.path.join(params.base_dir, f'featurized_{params.dataset}_pixels.pt'))

    # --- Run Test for Each Class ---
    results_per_class = {}
    print(f"\nRunning {params.my_test} test for each of the {params.num_classes} classes...")

    cifar10_classes = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]

    for c in range(params.num_classes):
        print(f"\n  Class {c} ({cifar10_classes[c]}):")
        real_c = real_samples_processed[c]
        fake_c = fake_samples_processed[c]

        if real_c.size(0) == 0 or fake_c.size(0) == 0:
            print(f"    Warning: Not enough samples for class {c} (Real: {real_c.size(0)}, Fake: {fake_c.size(0)}). Skipping.")
            results_per_class[c] = {'statistic': float('nan'), 'p_value': float('nan')}
            continue

        test_result = None
        p_value = None

        if params.my_test == 'mmd':
            test_result, p_value = mmd_test(real_c, fake_c)
        elif params.my_test == 'knn':
            test_result, p_value = knn_test(real_c, fake_c) 
        elif params.my_test == 'neural':
            test_result, p_value = neural_test(real_c, fake_c, epochs=params.neural_epochs, h=params.neural_hidden)
        else:
            print(f"    Error: Unknown test type '{params.my_test}'")
            test_result, p_value = float('nan'), float('nan')

        results_per_class[c] = {'statistic': test_result, 'p_value': p_value}
        print(f"    Result for Class {c}: Statistic = {test_result:.4f}, P-value = {p_value:.4f}")

    print("\n" + "="*40)
    print("--- Summary of Results Per Class ---")
    print("="*40)
    for c in range(params.num_classes):
        res = results_per_class[c]
        class_name = cifar10_classes[c] if c < len(cifar10_classes) else f"Class {c}"
        print(f"{class_name}: {params.my_test} Statistic = {res['statistic']:.4f}, P-value = {res['p_value']:.4f}")
    print("="*40)

    # You can further save results_per_class to a file (e.g., JSON, CSV) if needed
    # import json
    # with open(f"evaluation_results_{params.my_test}.json", "w") as f:
    #     json.dump(results_per_class, f, indent=4)


if __name__ == '__main__':
    main()