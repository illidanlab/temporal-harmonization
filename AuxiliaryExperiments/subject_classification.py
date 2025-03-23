import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from tools.utils import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import random
from Models.rnn import *

dic_pd_index = {"Sex":'nac_sex', "Edu":'nac_educ', "Age":'nac_a1_age'}
dic_attr_index = {"Age":["75-80", "81-87", "88-94"], "Edu":["12-15", "16-18", "19-21"], "Sex":[1, 2]}

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# Pad sequences
def pad_sequences(sequences):
    lengths = torch.tensor([len(seq) for seq in sequences])
    padded_seqs = nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    return padded_seqs, lengths

def mask_sequence(sequence, mask_percentage=70):
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

    return masked_sequence

def split_sequence(sequence, num_subsequences):
    subseq_length = len(sequence) // num_subsequences
    subsequences = [sequence[i * subseq_length:(i + 1) * subseq_length] for i in range(num_subsequences - 1)]
    subsequences.append(sequence[(num_subsequences - 1) * subseq_length:])
    
    return subsequences

class CustomDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = np.array(Y)
 
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
    
def collate_fn(batch):
    data, labels = zip(*batch)
    data = [torch.tensor(seq, dtype=torch.float32) for seq in data]
    labels = torch.tensor(np.array(labels), dtype=torch.float32)
    data_padded = pad_sequence(data, batch_first=True)
    return data_padded, labels

# Data Preprocess
dic_id2feature, df_labels, nl_subject, mci_subject = load_raw_data()

for task in ["Subject", "Age", "Edu", "Sex"]:
    set_seed(42)
    print(task)
        
    x, y = [], []
    all_features = []
    subject_id = 0

    for id in dic_id2feature:
        sequences = []
  
        for feature in dic_id2feature[id]:
            all_features.append(feature)
            sequences.append(feature)
        
        # Remove samples with little conversations
        if len(sequences) < 60:
            continue 
        
        for i in range(len(sequences)//5):
            x.append(mask_sequence(sequences, 80))
            if task == "Subject":
                y.append(subject_id)
            else:
                value = int(df_labels[df_labels["ts_sub_id"] == id][dic_pd_index[task]].values[0])
                for i in range(len(dic_attr_index[task])):
                    attr = dic_attr_index[task][i]
                    [lower, upper] = [int(attr[:2]), int(attr[3:])] if task != "Sex" else [attr, attr]
                    if value >= lower and value <= upper:
                        y.append(i)
                        break
        subject_id += 1

    # Hyperparameters
    input_dim = 99
    num_epochs = 200
    batch_size = 16
    learning_rate = 0.001

    scaler = StandardScaler()
    scaler = scaler.fit(all_features)

    for i in range(len(x)):
        x[i] = scaler.transform(x[i])

    accs = []

    for seed in range(100):
        set_seed(seed)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed, stratify=y)

        # Data loaders
        dataloader_train = DataLoader(CustomDataset(x_train, y_train), batch_size = 128, shuffle = True, collate_fn=collate_fn)
        dataloader_test = DataLoader(CustomDataset(x_test, y_test), batch_size = 128, shuffle = True, collate_fn=collate_fn)

        # Model, loss function, and optimizer
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if task == "Subject":
            model = simpleRNN(input_dim, subject_id, name = "GRU", bidirectional=True).to(device)
        elif task == "Sex":
            model = simpleRNN(input_dim, 2, name = "GRU", bidirectional=True).to(device)
        else:
            model = simpleRNN(input_dim, 3, name = "GRU", bidirectional=True).to(device)        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Training the model
        model.train()
        for epoch in range(num_epochs):
            loss_epoch = []
            for x_batch, y_batch in dataloader_train:
                x_batch, y_batch = x_batch.float().to(device), y_batch.long().to(device)
                # Forward pass
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_epoch.append(loss.item())

        # Evaluate the model
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for x_batch, y_batch in dataloader_test:
                x_batch, y_batch = x_batch.float().to(device), y_batch.to(device)
                outputs = model(x_batch)
                _, predicted = torch.max(outputs, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

            accs.append(correct / total)

    print(np.mean(accs), np.std(accs))