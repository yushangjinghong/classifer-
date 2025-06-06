import torch
from scipy.stats import norm # For cdf
import math

def distances(x, y):
    """
    Computes the squared Euclidean distances between all pairs of rows in x and y.
    Equivalent to Lua's `distances` function.
    """
    # x: (N, D), y: (M, D)
    # result: (N, M)
    result = x.new_full((x.size(0), y.size(0)), 0) # Initialize with zeros
    
    # (x - y)^2 = x^2 - 2xy + y^2
    # x.pow(2).sum(1) : (N,) -> (N,1)
    # y.pow(2).sum(1) : (M,) -> (M,1)
    
    # -2 * x @ y.T : (N, M)
    result.addmm_(x, y.t(), beta=1, alpha=-2)
    
    # Add x^2 terms: (N,1) broadcast to (N,M)
    result.add_(x.pow(2).sum(dim=1, keepdim=True))
    
    # Add y^2 terms: (1,M) broadcast to (N,M)
    result.add_(y.pow(2).sum(dim=1, keepdim=True).t())
    
    return result

def rbf_kernel(x, y, gamma):
    """
    Computes the RBF kernel matrix K(x,y) = exp(-gamma * ||x-y||^2).
    Equivalent to Lua's `rbf` function, where gamma = 1.0 / (2 * g^2)
    """
    if gamma <= 0:
        raise ValueError("Gamma for RBF kernel must be positive.")
    
    sq_dist = distances(x, y)
    # Original Lua: `d = -1.0/(2*(g^2))`, `rbf(x,y,g)`. So gamma = 1 / (2*g^2)
    # In my RBF, I use `gamma`. So -gamma * ||x-y||^2.
    # The Lua `g` is `std`, so `d = -1.0 / (2 * std^2)`.
    # My `gamma` here should be `1.0 / (2 * std^2)` to match.
    # The original Lua code: `rbf(x,y,g)`. Here `g` is `std`.
    # The calculation is `torch.exp(torch.mul(torch.sum(torch.pow(x-y,2),2),d))`
    # where `d = -1.0/(2*(g^2))`.
    # So, `exp(-1.0/(2*g^2) * sum(pow(x-y,2)))`
    # This means the `gamma` parameter in my `rbf_kernel` should be `1.0/(2*g^2)` from the Lua `g`.
    # Let's adjust this for clarity. The Lua `g` is a standard deviation-like parameter.
    # `gamma` in common ML RBF kernels is usually `1/ (2 * sigma^2)`.
    # So if Lua's `g` is `sigma`, then `gamma = 1.0 / (2 * g**2)`.
    
    # Ensure sq_dist is non-negative, can happen with floating point errors
    sq_dist[sq_dist < 0] = 0 
    
    return torch.exp(-gamma * sq_dist) # Using common gamma definition

def mmd_stat(x, y, g_std):
    """
    Calculates the MMD statistic and its variance.
    `g_std` here corresponds to the 'g' (std) parameter from Lua's `rbf`.
    """
    m = x.size(0)
    if (m % 2) == 1: # ensure m is even
        m -= 1
    m2 = math.ceil(m / 2.0) # Using math.ceil for consistency with Lua
    m2_int = int(m2) # Convert to int for slicing
    
    # Calculate gamma from g_std for rbf_kernel
    gamma = 1.0 / (2 * (g_std**2))

    # Split data for U-statistic
    x1 = x[:m2_int]
    x2 = x[m2_int:m]
    y1 = y[:m2_int]
    y2 = y[m2_int:m]

    kxx = rbf_kernel(x1, x2, gamma)
    kyy = rbf_kernel(y1, y2, gamma)
    kxy = rbf_kernel(x1, y2, gamma)
    kyx = rbf_kernel(x2, y1, gamma) # This is `rbf(x[{{m2+1,m}}], y[{{1,m2}}]` in Lua

    res = kxx + kyy - kxy - kyx
    
    # Ensure inputs are float/double
    res = res.float() if res.dtype == torch.float64 else res # Ensure it's float for mean/var
    
    mean = res.mean()
    variance = res.var() / m2 # As per Lua code's `res:var()/m2`
    
    return mean, variance

def choose_g(x, y):
    """
    Heuristically chooses the best 'g' (std) parameter for the RBF kernel.
    """
    # Powers of 2 from -15 to 10
    g_values = torch.pow(2, torch.arange(-15.0, 11.0))
    epsilon = 1e-4 # Small constant for stability (Lua's `l`)
    
    m2 = math.ceil(x.size(0) / 2.0)
    
    best_g = 1.0
    best_ratio = 0.0

    for g_val in g_values:
        # Use a subset of data for choosing g, as in the Lua code.
        # This implies mmd_stat takes `x[{{1,m2}}],y[{{1,m2}}]` as its first two arguments.
        # However, mmd_stat internally splits these further.
        # The Lua code for `choose_g` uses `x[{{1,m2}}],y[{{1,m2}}]`
        # as arguments to `mmd_stat`. Let's clarify this.
        # The Lua `mmd_stat` function takes `x` and `y` and splits *them* into `x1,x2,y1,y2`.
        # So `choose_g(x_full, y_full)` should pass `x_full[:m2_for_g_choice]`, `y_full[:m2_for_g_choice]`
        # to `mmd_stat` which will then split them again. This seems redundant or misunderstood.
        # A more standard approach for `choose_g` (median heuristic) is often used.
        # Given the explicit Lua `mmd_stat(x[{{1,m2}}],y[{{1,m2}}],g[i])`, it suggests
        # taking the first half of the data for `choose_g`.

        # Let's simplify this. `mmd_stat` needs enough data to split itself.
        # If `choose_g` is fed the full `x` and `y`, it will internally split them.
        # So `x_slice = x[:int(m2)]`, `y_slice = y[:int(m2)]`.
        # Then `mmd_stat(x_slice, y_slice, g_val)`.
        
        # To strictly follow Lua:
        # x_for_g = x[:int(m2_main_script)] # Assuming m2 from outer scope
        # y_for_g = y[:int(m2_main_script)]
        # However, `mmd_stat` itself does `m = x:size(1)`, so it finds its own `m` and `m2`.
        # So, simply pass `x` and `y` to `mmd_stat` for choosing `g`.
        
        # To avoid re-calculating `m2` and splitting multiple times,
        # it's common to use a fixed subset for `choose_g` or a simpler heuristic.
        
        # The Lua code explicitly calculates `m2` in `choose_g` and uses it.
        # Let's use `x` and `y` directly as input to `choose_g` as per the Lua `mmd_test` call.
        # `mmd_test` takes `x`, `y` and passes `x[{{1,m2}}],y[{{1,m2}}]` to `choose_g`.
        
        # So, the `x` and `y` passed to `choose_g` are already a subset from `mmd_test`.
        # The `mmd_stat` inside `choose_g` will then further split these subsets.
        # This seems like nested splitting.
        # Lua: `g = choose_g(x[{{1,m2}}],y[{{1,m2}}])`
        # So, `x` and `y` coming into `choose_g` are already `x_first_half` and `y_first_half`.
        # `mmd_stat` then splits *these* into their own first and second halves.
        
        # Let's calculate the `m2` that `mmd_stat` will use internally if given `x` and `y`.
        current_m_for_stat = x.size(0)
        if (current_m_for_stat % 2) == 1: current_m_for_stat -= 1
        internal_m2_for_stat = math.ceil(current_m_for_stat / 2.0)
        
        # To truly follow the Lua behavior, we need to pass the arguments as Lua did.
        # Lua: `mmd_m, mmd_v = mmd_stat(x[{{m2+1,m}}],y[{{m2+1,m}}],g)`
        # This means `choose_g` is called with `x[:m2_overall]`, `y[:m2_overall]`
        # and `mmd_stat` is called with `x[m2_overall:]`, `y[m2_overall:]`.
        
        # This implies `choose_g` calculates its own `mmd_stat` using *its* inputs.
        # Let's assume `x` and `y` given to `choose_g` are the relevant parts.
        
        mean_stat, variance_stat = mmd_stat(x, y, g_val) # x and y here are *already* the first halves
        
        # Avoid division by zero if variance is too small
        denominator = torch.sqrt(variance_stat * internal_m2_for_stat) + epsilon
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
    m = x.size(0)
    if (m % 2) == 1:
        m -= 1 # Ensure m is even
    m2 = math.ceil(m / 2.0) # Overall m2 for splitting x, y into two halves
    m2_int = int(m2)

    # Choose 'g' using the first half of the data, as per Lua
    g = choose_g(x[:m2_int], y[:m2_int])

    # Calculate MMD statistic and variance using the second half of the data
    mmd_mean, mmd_variance = mmd_stat(x[m2_int:m], y[m2_int:m], g)
    
    # Calculate p-value. Assumes mmd_mean is normally distributed N(0, sqrt(mmd_variance)).
    # The CDF is for the standard normal distribution. If mmd_mean is not centered at 0,
    # then it's a shifted normal distribution.
    # P(Z > z) = 1 - CDF(z)
    # The Lua code uses `distributions.norm.cdf(mmd_m,0,math.sqrt(mmd_v))`
    # which is `cdf(value, mean, std_dev)`.
    # So `norm.cdf(mmd_mean, 0, math.sqrt(mmd_variance))`.
    
    # Handle very small variance to avoid `nan` or `inf`
    if mmd_variance <= 0:
        p_value = 1.0 if mmd_mean <= 0 else 0.0 # If variance is 0, p-value is 1 if mean non-positive, else 0
    else:
        std_dev = math.sqrt(mmd_variance)
        # Assuming the null hypothesis is that MMD is 0.
        # We want P(MMD > observed_MMD) = 1 - CDF(observed_MMD | mean=0, std=std_dev)
        p_value = 1.0 - norm.cdf(mmd_mean.item(), loc=0, scale=std_dev)
    
    return mmd_mean.item(), p_value

if __name__ == '__main__':
    # Example usage for mmd_test
    # Generate some synthetic data
    torch.manual_seed(42)
    
    # Two distributions that are slightly different
    x_data = torch.randn(200, 10) * 0.5 + 1.0
    y_data = torch.randn(200, 10) * 0.5 + 1.2
    
    # Test identical distributions
    print("Testing MMD with identical distributions:")
    stat_same, p_same = mmd_test(x_data, x_data)
    print(f"MMD Stat: {stat_same:.6f}, P-value: {p_same:.6f}") # Should be close to 0 and 1
    
    # Test different distributions
    print("\nTesting MMD with different distributions:")
    stat_diff, p_diff = mmd_test(x_data, y_data)
    print(f"MMD Stat: {stat_diff:.6f}, P-value: {p_diff:.6f}") # Should be higher and smaller
    
    # Test with very few samples (can lead to unstable results)
    x_small = torch.randn(10, 5)
    y_small = torch.randn(10, 5) + 0.1
    print("\nTesting MMD with small number of samples:")
    stat_small, p_small = mmd_test(x_small, y_small)
    print(f"MMD Stat: {stat_small:.6f}, P-value: {p_small:.6f}")