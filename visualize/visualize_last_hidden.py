# Visualize

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from tools.utils import *
from sklearn.preprocessing import StandardScaler
import random
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

colors = [
    "#1f77b4",  # Blue
    "#ff7f0e",  # Orange
    "#2ca02c",  # Green
    "#d62728",  # Red
    "#9467bd",  # Purple
    "#8c564b",  # Brown
    "#e377c2",  # Pink
    "#7f7f7f",  # Gray
    "#bcbd22",  # Yellow-green
    "#17becf",  # Cyan
    "#ffbb78",  # Light Orange
    "#98df8a",  # Light Green
    "#ff9896",  # Light Red
    "#c5b0d5",  # Light Purple
    "#c49c94"   # Light Brown
]

# Data Preprocess
dic_id2feature, df_labels, nl_subject, mci_subject = load_raw_data()

def split_sequence(sequence, num_subsequences):
    subseq_length = len(sequence) // num_subsequences
    subsequences = [sequence[i * subseq_length:(i + 1) * subseq_length] for i in range(num_subsequences - 1)]
    subsequences.append(sequence[(num_subsequences - 1) * subseq_length:])
    
    return subsequences

def mask_sequence(sequence, mask_percentage=50):
    # Calculate the number of elements to mask
    num_elements_to_mask = int(len(sequence) * mask_percentage / 100)
    
    # Generate random indices to mask
    indices_to_mask = random.sample(range(len(sequence)), num_elements_to_mask)

    # Create a copy of the sequence to avoid modifying the original
    masked_sequence = []

    # Mask the selected indices
    for i in range(len(sequence)):
        if i not in indices_to_mask:
            masked_sequence.append(sequence[i])

    return np.array(masked_sequence)

def generate_masked_sequences(x, y, mask_percentage=75):
    masked_sequences = []
    cog_label = []
    sbj_label = []
    for i in range(len(x)):
        for seed in range(len(x[i])):
            random.seed(seed)
            masked_sequences.append(mask_sequence(x[i], mask_percentage))
            cog_label.append(y[i])
            sbj_label.append(i)

    return masked_sequences, cog_label, sbj_label

def plot(name = "before", model_name = "GRU"):
    if name == "before":
        harmonized_model = torch.load(f"weights/temporal_solver_{model_name}.pt",weights_only=False).to("cuda")
    else:
        harmonized_model = torch.load(f"weights/temporal_sequence_harmonization_adversarial_solver_{model_name}.pt",weights_only=False).to("cuda") # harmonization

    set_seed(42)

    x, y = [], []
    all_features = []
    subject_id = 0

    for id in dic_id2feature:
        sequences = []

        for feature in dic_id2feature[id]:
            all_features.append(feature)
            sequences.append(feature)
        
        if len(sequences) < 60:
            continue 
        
        for i in range(len(sequences)//5):
            x.append(mask_sequence(sequences, 80))
            y.append(subject_id)

        subject_id += 1

    scaler = StandardScaler()
    scaler = scaler.fit(all_features)
    all_features = scaler.transform(all_features)

    x_new = []
    y_new = []

    for i in range(len(x)):
        x[i] = scaler.transform(x[i])

        out = harmonized_model(torch.tensor([x[i]]).float().to("cuda"), id = "3") #
        
        x_new.append(out.detach().cpu().numpy()[0])
        y_new.append(y[i])

    x_new = TSNE(n_components=2, random_state=42).fit_transform(np.array(x_new))
    y_new = np.array(y_new)

    # Plot the reduced dimension data for each sample
    plt.figure(figsize=(12, 8))

    c = 0
    for i in range(15):
        plt.scatter(x_new[y_new == i, 0], x_new[y_new == i, 1], label=f'Subject {c}', c=colors[c], alpha=0.5)
        c += 1

    # plt.title('Plot GRU last hidden state of each subject subsequence')
    plt.xlabel('TSNE Component 1', fontsize = 15)
    plt.ylabel('TSNE Component 2', fontsize = 15)
    plt.grid(True)

    if name == "after":
        plt.legend(loc='upper right', bbox_to_anchor=(1.14, 1), frameon=False)  # 1.15 moves the legend outside

    plt.savefig(f"figures/{name}_harmonization_{model_name}.pdf", dpi = 300)
    plt.savefig(f"figures/{name}_harmonization_{model_name}.png", dpi = 300)

    plt.clf()

for model_name in ["GRU"]:
    plot("before", model_name=model_name)
    plot("after", model_name=model_name)