import warnings
# Suppress all warnings for cleaner output during execution
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.ensemble import RandomForestClassifier as RFC
from scipy.interpolate import UnivariateSpline as sp
from sklearn.preprocessing import scale
from sklearn.mixture import GaussianMixture # Changed GMM to GaussianMixture
from sklearn.linear_model import LogisticRegression as LR # Although LR is imported, it's not used in the provided code
import pickle
from tqdm import tqdm

def rp(k,s,d):
    """Generates random projection weights."""
    return np.hstack((np.vstack([si*np.random.randn(k,d) for si in s]),
                      2*np.pi*np.random.rand(k*len(s),1))).T

def f1(x,w):
    """Applies a cosine transformation with weights."""
    return np.cos(np.dot(np.hstack((x,np.ones((x.shape[0],1)))),w))

def f2(x,y,z):
    """Combines features from x, y, and concatenated x,y."""
    return np.hstack((f1(x,wx).mean(0),f1(y,wy).mean(0),f1(z,wz).mean(0)))

# ... (rest of your imports and code) ...

def cause(n,k,p1,p2):
    """Generates 'cause' data using a Gaussian Mixture Model."""
    g = GaussianMixture(n_components=k)
    g.means_ = p1*np.random.randn(k,1)
    g.covariances_ = np.power(abs(p2*np.random.randn(k,1)+1),2).reshape(k, 1, 1) # Reshape for GaussianMixture

    # --- THIS IS THE FIX ---
    # Ensure weights_ is a 1-dimensional array
    g.weights_ = abs(np.random.rand(k)) # Create a 1D array directly
    g.weights_ = g.weights_ / np.sum(g.weights_) # Normalize

    # Alternatively, if you prefer the 2D creation for some reason, flatten it:
    # g.weights_ = abs(np.random.rand(k, 1)).flatten()
    # g.weights_ = g.weights_ / np.sum(g.weights_)
    # -----------------------

    return scale(g.sample(n)[0]) # sample returns a tuple (X, y)

# ... (rest of your code) ...

def noise(n,v):
    """Generates random noise."""
    return v*np.random.rand(1)*np.random.randn(n,1)

def mechanism(x,d):
    """Applies a non-linear mechanism to the input data."""
    # Ensure x is 1D for UnivariateSpline
    x_flat = x.flatten()
    g = np.linspace(np.min(x_flat)-np.std(x_flat), np.max(x_flat)+np.std(x_flat), d)
    # Using s=0 for interpolation that goes through all points
    return sp(g, np.random.randn(d), s=0)(x_flat)[:,np.newaxis]

def pair(n=1000,k=3,p1=2,p2=2,v=2,d=5):
    """Generates a causal pair (cause and effect)."""
    x = cause(n,k,p1,p2)
    return (x, scale(scale(mechanism(x,d)) + noise(n,v)))

def pairset(N):
    """Generates a dataset of causal pairs and their reversed counterparts."""
    z1 = np.zeros((N,3*wx.shape[1]))
    z2 = np.zeros((N,3*wx.shape[1]))
    print(f"Generating {N} synthetic pairs...")
    for i in tqdm(range(N)):
        (x,y) = pair()
        z1[i,:] = f2(x,y,np.hstack((x,y)))
        z2[i,:] = f2(y,x,np.hstack((y,x)))
    return (np.vstack((z1,z2)),np.hstack((np.zeros(N),np.ones(N))).ravel(),np.ones(2*N))

def tuebingen(fname = 'tuebingen.pkl'):
    """Loads and processes the Tuebingen dataset."""
    try:
        with open(fname, 'rb') as f: # Changed 'r' to 'rb' for binary reading
            x,y,w = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: {fname} not found. Please run the data preparation script first.")
        return None, None, None
    except Exception as e:
        print(f"Error loading pickle file {fname}: {e}")
        return None, None, None

    print(f"Loaded {len(x)} samples from {fname}")

    z = np.zeros((len(x),3*wx.shape[1]))
    print("Processing Tuebingen dataset features...")
    for i in tqdm(range(len(x))): # Changed xrange to range
        a = scale(x[i][:,0])[:,np.newaxis]
        b = scale(x[i][:,1])[:,np.newaxis]
        z[i,:] = f2(a,b,np.hstack((a,b)))

    return z,y,w

np.random.seed(0)

# Constants for model training
N = 10000 # Number of synthetic pairs
K = 333   # Number of random projection features (controls complexity of f1)
E = 500   # Number of estimators for Random Forest Classifier

# Random projection weights
# These are global variables used by f1 and f2
wx = rp(K,[0.15,1.5,15],1)
wy = rp(K,[0.15,1.5,15],1)
wz = rp(K,[0.15,1.5,15],2)

print("Generating synthetic training and test datasets...")
# Generate synthetic training data
(x1,y1,m1) = pairset(N)
# Generate synthetic test data
(x2,y2,m2) = pairset(N)
print("Synthetic datasets generated.")

print("Loading and processing Tuebingen dataset...")
# Load and process Tuebingen real-world data
(x0,y0,m0) = tuebingen()

if x0 is None:
    print("Exiting due to data loading error.")
else:
    print("Training Random Forest Classifier...")
    # Train the Random Forest Classifier
    # n_jobs=-1 uses all available CPU cores
    reg = RFC(n_estimators=E,random_state=0,n_jobs=-1).fit(x1,y1)
    print("Random Forest Classifier trained.")

    # Evaluate the model and print scores
    print(f"Results: [N={N}, K={K}, E={E}, "
          f"Train Score (Synthetic): {reg.score(x1,y1,m1):.4f}, "
          f"Test Score (Synthetic): {reg.score(x2,y2,m2):.4f}, "
          f"Tuebingen Score (Real-world): {reg.score(x0,y0,m0):.4f}]")