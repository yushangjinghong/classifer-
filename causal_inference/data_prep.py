import pickle
import numpy as np
from subprocess import call
import os # Import os for path manipulation

# Download and unzip the dataset
print("Downloading and unzipping data...")
call(["wget", "https://webdav.tuebingen.mpg.de/cause-effect/pairs.zip"])
call(["unzip", "-d", "pairs", "pairs.zip"])
call(["rm", "pairs.zip"])
print("Data downloaded and unzipped.")

# Load metadata
meta = np.genfromtxt('pairs/pairmeta.txt', dtype=str) # Use dtype=str for consistency

weights = []
samples = []
labels  = []

print("Processing data pairs...")
for i in range(meta.shape[0]): # Changed xrange to range
    d = np.genfromtxt(os.path.join('pairs', 'pair' + meta[i][0] + '.txt')) # Use os.path.join for robust path handling
    x = d[:,0]
    y = d[:,1]

    # Conditions for appending data
    if((meta[i][1] == '1') and
       (meta[i][2] == '1') and
       (meta[i][3] == '2') and
       (meta[i][4] == '2')):
        samples.append(np.vstack((x,y)).T)
        labels.append(0)
        weights.append(float(meta[i][5]))
        samples.append(np.vstack((y,x)).T)
        labels.append(1)
        weights.append(float(meta[i][5]))

    if((meta[i][1] == '2') and
       (meta[i][2] == '2') and
       (meta[i][3] == '1') and
       (meta[i][4] == '1')):
        samples.append(np.vstack((y,x)).T)
        labels.append(0)
        weights.append(float(meta[i][5]))
        samples.append(np.vstack((x,y)).T)
        labels.append(1)
        weights.append(float(meta[i][5]))

# Save processed data to a pickle file
try:
    with open('tuebingen.pkl', 'wb') as f: # Changed 'w' to 'wb' for binary writing
        pickle.dump((samples, labels, weights), f)
    print("Data successfully saved to tuebingen.pkl")
except Exception as e:
    print(f"Error saving pickle file: {e}")

# Clean up temporary files
call(["rm", "-rf", "pairs"])
print("Temporary 'pairs' directory removed.")