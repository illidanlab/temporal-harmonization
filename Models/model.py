import torch
from torch.utils.data import Dataset
import numpy as np

# Define model
class MLP_pytorch(torch.nn.Module):
    def __init__(self, input_dim, output_dim, model_name = "MLP"):
        super(MLP_pytorch, self).__init__()

        self.model_name = model_name

        if self.model_name == "LR":
            # LR
            self.dropout = torch.nn.Dropout(0.3)
            self.linear1 = torch.nn.Linear(input_dim, output_dim)
        else:
            # MLP
            self.linear1 = torch.nn.Linear(input_dim, 32)
            self.relu1 = torch.nn.ReLU()
            self.linear2 = torch.nn.Linear(32, output_dim)
    
    def forward(self, x):
        if self.model_name == "LR":
            # LR
            return self.linear1(self.dropout(x))

        # MLP
        outputs = self.linear1(x)
        outputs = self.relu1(outputs)
        outputs = self.linear2(outputs)
        return outputs
    
class LR_pytorch(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LR_pytorch, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, output_dim)
    def forward(self, x):
        return self.linear1(x) 

class MSE_pytorch(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MSE_pytorch, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, output_dim)
    def forward(self, x):
        return self.linear1(x) 
    
# Subject Harmonization: https://pubmed.ncbi.nlm.nih.gov/38160279/
class MLP_harmonization(torch.nn.Module):
    feature_idx = None
    def __init__(self, input_dim, sbj_dim, task_in_dim, task_out_dim, model_name = "MLP"):
        super(MLP_harmonization, self).__init__()
        self.feature_mapping = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, input_dim),
        )
            
        self.out_sbj = torch.nn.Linear(input_dim, sbj_dim)
        self.model_name = model_name
        if self.model_name == "LR":
            self.out_task = torch.nn.Sequential(
                # LR
                torch.nn.Linear(task_in_dim, task_out_dim),
            )
        else:
            self.out_task = torch.nn.Sequential(
                # MLP
                torch.nn.Linear(task_in_dim, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, task_out_dim)
            )
    
    def forward(self, x, id):
        feature = self.feature_mapping(x)
        if id == "0":
            return feature
        elif id == "1":
            return self.out_sbj(feature)
        elif id == "0,1":
            return [feature, self.out_sbj(feature)]
        elif id == "0,1,2":
            return [feature, self.out_sbj(feature), self.out_task(feature)]
        elif id == "2":
            if self.feature_idx is not None:
                return self.out_task(feature[:, self.feature_idx])
            else:
                return self.out_task(feature)
            
class CustomDataset(Dataset):
    def __init__(self, X, Y, G):
        self.X = X
        self.Y = np.array(Y)
        self.G = np.zeros(len(G), dtype=np.int64)
        self.subject_id = {}
        g_unique = list(sorted(set(G)))
        for i, g in enumerate(g_unique):
            index = [i for i in range(len(G)) if G[i] == g]
            self.G[index] = i
            self.subject_id[i] = g
 
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.G[idx], idx
    
    def kept(self, idx_kept):
        self.X = self.X[idx_kept]
        self.Y = self.Y[idx_kept]
        self.G = self.G[idx_kept]
        return self

class CustomDatasetGroup(Dataset):
    def __init__(self, X, Y, G):
        self.X = X
        self.Y = np.array(Y)
        self.G = np.array(G)
 
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.G[idx], idx
    
    def kept(self, idx_kept):
        self.X = self.X[idx_kept]
        self.Y = self.Y[idx_kept]
        return self