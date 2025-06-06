print("脚本开始执行！")
import numpy as np
import torch
import os
import sys # Keep sys if you need to manage paths, but it's not directly related to the pickle issue here

# Define the base directory for your data
data_base_dir = '/home/yushangjinghong/Desktop/classifer_l/interpretable-test/freqopttest/data'
output_dir = '.' # Save .pt files in the current directory, or specify another path

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

documents = [
    "bayes_bayes_np430_nq432_d2000.p",
    "bayes_deep_np846_nq433_d2000_random_noun.p",
    "bayes_learning_np821_nq276_d2000.p",
    "bayes_neuro_np794_nq788_d2000_random_noun.p",
    "deep_neuro_np105_nq512_d2000.p",
    "deep_learning_np431_nq299_d2000_random_noun.p",
    "neuro_learning_np832_nq293_d2000.p"
]

print("Processing document datasets...")
for doc_name in documents:
    print(f"Loading: {doc_name}")
    
    # Construct the full path to the .p file
    file_path = os.path.join(data_base_dir, doc_name)
    
    # --- THE CRUCIAL CHANGE IS HERE ---
    # Load the pickled NumPy data, specifying encoding for Python 2 compatibility
    x = np.load(file_path, allow_pickle=True, encoding='latin1')
    # --- END OF CRUCIAL CHANGE ---
    
    # Extract 'P' and 'Q' arrays
    p_np = x['P']
    q_np = x['Q']
    
    # Convert NumPy arrays to PyTorch tensors (and ensure float32 for consistency)
    p_tensor = torch.from_numpy(p_np).float()
    q_tensor = torch.from_numpy(q_np).float()
    
    # Determine the base name for the output .pt file (e.g., 'bayes_bayes')
    output_base_name = '_'.join(doc_name.split('_')[:2])
    output_file_name = os.path.join(output_dir, f"{output_base_name}.pt")
    
    # Save as a PyTorch .pt file containing a dictionary with 'p' and 'q' tensors
    torch.save({'p': p_tensor, 'q': q_tensor}, output_file_name)
    print(f"Saved {output_file_name}")

print("\nProcessing faces datasets...")

# --- Faces Different ---
faces_diff_path = os.path.join(data_base_dir, 'crop48_HANESU_AFANDI.p')
print(f"Loading: {faces_diff_path}")
# --- THE CRUCIAL CHANGE IS HERE ---
faces_diff = np.load(faces_diff_path, allow_pickle=True, encoding='latin1')
# --- END OF CRUCIAL CHANGE ---

# Extract X and Y (assuming they are directly attributes of the loaded object)
p_diff_tensor = torch.from_numpy(faces_diff.X).float()
q_diff_tensor = torch.from_numpy(faces_diff.Y).float()

output_file_name_diff = os.path.join(output_dir, 'faces_diff.pt')
torch.save({'p': p_diff_tensor, 'q': q_diff_tensor}, output_file_name_diff)
print(f"Saved {output_file_name_diff}")

# --- Faces Same ---
faces_same_path = os.path.join(data_base_dir, 'crop48_h0.p')
print(f"Loading: {faces_same_path}")
# --- THE CRUCIAL CHANGE IS HERE ---
faces_same = np.load(faces_same_path, allow_pickle=True, encoding='latin1')
# --- END OF CRUCIAL CHANGE ---

# For 'faces_same', the description implies both p and q come from the same source
p_same_tensor = torch.from_numpy(faces_same).float()
q_same_tensor = torch.from_numpy(faces_same).float()

output_file_name_same = os.path.join(output_dir, 'faces_same.pt')
torch.save({'p': p_same_tensor, 'q': q_same_tensor}, output_file_name_same)
print(f"Saved {output_file_name_same}")

print("\nData preprocessing complete. Your .pt files are ready.")