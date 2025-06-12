import numpy as np
from Solvers.Solver_Base import Solver_Base
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from Models.rnn import RNN_temporal_harmonization
import random
from torch.utils.data import Dataset
from tools.utils import permute_feature
import os

class CustomDataset(Dataset):
    def __init__(self, X, Y, G):
        self.X = X
        self.Y = np.array(Y)
        self.G = np.array(G)
 
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.G[idx]

def collate_fn(batch):
    data, labels, g = zip(*batch)
    data = [torch.tensor(seq, dtype=torch.float32) for seq in data]
    labels = torch.tensor(labels, dtype=torch.float32)
    g = torch.tensor(g, dtype = torch.float32)
    data_padded = pad_sequence(data, batch_first=True)
    return data_padded, labels, g

def split_subsequences(arr):
    subsequences = []
    for i in range(len(arr)):
        subsequences.append(arr[:i+1])
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

def generate_masked_sequences_test(x, y, g, mask_percentage=0, length_subsequence = -1):  
    if mask_percentage == 0 and length_subsequence == -1:
        return x, y, g
    
    masked_sequences = []
    cog_label = []
    g_sequences = []
    for i in range(len(x)):
        random.seed(0)
        if mask_percentage != 0:
            masked_sequences.append(mask_sequence(x[i], mask_percentage))
            cog_label.append(y[i])
            g_sequences.append(g[i])
        elif length_subsequence != -1 and len(x[i]) > length_subsequence:
            masked_sequences.append(mask_sequence(x[i], 100*(len(x[i]) - length_subsequence)/len(x[i])))
            cog_label.append(y[i])
            g_sequences.append(g[i])

    return masked_sequences, np.array(cog_label), np.array(g_sequences)

class temporal_sequence_harmonization_adversarial_solver(Solver_Base):
    
    def __init__(self, cfg_proj, cfg_m, name = "temporal_harmonization"):
        Solver_Base.__init__(self, cfg_proj, cfg_m, name)
        
    def run(self, x_train, y_train, g_train, x_test, y_test, g_test, seed):
    
        self.set_random_seed(seed)

        # Sequence Masking
        x_train_mask, y_train_mask, g_train_mask = generate_masked_sequences(x_train, y_train, mask_percentage=self.cfg_proj.mask)
        x_test, y_test, g_test = generate_masked_sequences_test(x_test, y_test, g_test, mask_percentage=self.cfg_proj.mask_test, length_subsequence=self.cfg_proj.test_length)

        random.seed(seed)
        
        dataloader_train = DataLoader(CustomDataset(x_train_mask, y_train_mask, g_train_mask), 
                                      batch_size = self.cfg_m.training.batch_size, drop_last=False, shuffle = True, pin_memory=True, worker_init_fn = np.random.seed(seed), collate_fn=collate_fn)
        
        model = RNN_temporal_harmonization(input_dim = len(x_train_mask[0][0]), sbj_dim = len(list(set(g_train_mask))), task_in_dim = len(x_train_mask[0][0]), task_out_dim = self.cfg_m.data.dim_out, adversarial=True, name=self.cfg_proj.model_name) 
        model = model.to(self.device)
        
        optimizer = None
        lr_scheduler = None
        model, loss_train_trace = self.train(model, dataloader_train, optimizer, lr_scheduler, epochs=self.cfg_m.training.epochs, alpha1 = self.cfg_m.alpha1, alpha2 = self.cfg_m.alpha2, alpha3 = self.cfg_m.alpha3)

        # Save Weight
        os.makedirs("weights", exist_ok=True)
        torch.save(model, f"weights/{self.cfg_proj.solver}_{self.cfg_proj.model_name}.pt")

        # Evaluation
        auc, f1, sens, spec, auc_sbj, f1_sbj, sens_sbj, spec_sbj = self.eval_func(model, x_test, y_test, g_test)
        
        # Feature Importance Experiment
        if self.cfg_proj.experiment == "FeatureImportance":
            self.AUC_permute_index = []
            for index in range(99):
                x_test_permute = permute_feature(x_test, index = index)
                auc_index, _, _, _, _, _, _, _ = self.eval_func(model, x_test_permute, y_test, g_test)
                self.AUC_permute_index.append(auc_index)

        return auc, f1, sens, spec, auc_sbj, f1_sbj, sens_sbj, spec_sbj

    def train(self, model, dataloader_train, optimizer, lr_scheduler, epochs = 100, stage = None, alpha1 = 0.5, alpha2 = 0.5, alpha3 = 0.1):
        loss_train_trace = []
        loss_mse = torch.nn.MSELoss()
        loss_cross_ent_sbj = torch.nn.CrossEntropyLoss()
        loss_cross_ent_cog = torch.nn.CrossEntropyLoss()
        
        optimizer_G = torch.optim.AdamW(list(model.feature_mapping.parameters()) + list(model.out_task.parameters()), lr = self.cfg_m.training.lr_init)
        optimizer_D = torch.optim.AdamW(model.out_sbj.parameters(), lr = self.cfg_m.training.lr_init/5)
        
        lr_scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_G, int(epochs*len(dataloader_train))) 
        lr_scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_D, int(epochs*len(dataloader_train)/2)) 
        
        for epoch in range(epochs):
            model.train()
            loss_epoch_G = []
            loss_epoch_D = []
            for train_X, train_Y, train_G in dataloader_train:   
                train_X, train_Y, train_G = train_X.float().to(self.device), train_Y.long().to(self.device), train_G.long().to(self.device)
                [features, logits_sbj, logits_cog] = model(train_X, id = "0,1,2") #
            
                loss_sbj_pos = loss_cross_ent_sbj(logits_sbj, train_G)
                
                # Trick for training Discriminator and Generator
                if epoch % 2 == 0:
                    loss_epoch_D.append(loss_sbj_pos.item())
                    model.zero_grad()
                    optimizer_D.zero_grad()
                    loss_sbj_pos.backward()
                    optimizer_D.step()
                    lr_scheduler_D.step()

                train_X, train_Y, train_G = train_X.float().to(self.device), train_Y.long().to(self.device), train_G.long().to(self.device)
                [features, logits_sbj, logits_cog] = model(train_X, id = "0,1,2") #

                # Temporal Tendency Regularization Loss
                loss_sbj_mse = loss_mse(train_X, features)

                # Adversarial Loss
                loss_sbj_neg =  -loss_cross_ent_sbj(logits_sbj, train_G)

                # Cognitive Classification Loss
                loss_cog = loss_cross_ent_cog(logits_cog, train_Y)

                # Temporal Smoothness Loss
                # first order
                differences = features[:, 1:, :] - features[:, :-1, :]
                regu1 = torch.mean(torch.norm(differences, p = 2, dim = 2))
                # second order
                differences = features[:, 2:, :] - features[:, 1:-1, :]*2 + features[:, :-2, :] 
                regu2 = torch.mean(torch.norm(differences, p = 2, dim = 2))
                
                loss = loss_cog +  (loss_sbj_neg * alpha1) + (loss_sbj_mse * alpha2) + (regu1*alpha3) + (regu2*alpha3)
        
                loss_epoch_G.append(loss.item())
                model.zero_grad()
                optimizer_G.zero_grad()
                loss.backward()
                optimizer_G.step()
                lr_scheduler_G.step()

            loss_train_trace.append(np.mean(loss_epoch_G))
        return model, loss_train_trace
    
    def predict(self, model, X, flag_prob = False):
        model.eval()
        with torch.no_grad():
            pred = []
            for seq in X:
                seq = torch.tensor(seq).unsqueeze(0).float().to(self.device)
                p = model(seq, id = "2")[0]
                pred.append(p.detach().cpu().numpy())
        pred = np.argmax(pred, 1)
      
        return pred

    def predict_proba(self, model, X, flag_prob = True):
        model.eval()
        with torch.no_grad():
            pred = []
            for seq in X:
                seq = torch.tensor(seq).unsqueeze(0).float().to(self.device)
                p = model(seq, id = "2")[0]
                pred.append(p.detach().cpu().numpy())
            pred = np.array(pred)  # Ensure pred is a NumPy array
            pred_max = np.max(pred, axis=1, keepdims=True)  # Get max per row
            pred = np.exp(pred - pred_max) / np.sum(np.exp(pred - pred_max), axis=1, keepdims=True)

        return pred