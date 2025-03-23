import numpy as np
from Solvers.Solver_Base import Solver_Base
import torch
from torch.utils.data import DataLoader
from Models.model import CustomDataset
from Models.rnn import simpleRNN
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import random
from tools.utils import permute_feature

def collate_fn(batch):
    data, labels, g, idx = zip(*batch)
    data = [torch.tensor(seq, dtype=torch.float32) for seq in data]
    labels = torch.tensor(np.array(labels), dtype=torch.float32)
    data_padded = pad_sequence(data, batch_first=True)
    return data_padded, labels, g, idx

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

class temporal_solver(Solver_Base):
    
    def __init__(self, cfg_proj, cfg_m, name = "temporal"):
        Solver_Base.__init__(self, cfg_proj, cfg_m, name)
        
    def run(self, x_train, y_train, g_train, x_test, y_test, g_test, seed):
        x_test, y_test, g_test = generate_masked_sequences_test(x_test, y_test, g_test, mask_percentage=self.cfg_proj.mask_test, length_subsequence=self.cfg_proj.test_length)

        x_test = [torch.tensor(seq, dtype=torch.float32) for seq in x_test]
        x_test = pad_sequence(x_test, batch_first=True)    
        
        # Initialize
        dataloader_train = DataLoader(CustomDataset(x_train, y_train, g_train), batch_size = 8, drop_last=True, shuffle = True, collate_fn=collate_fn)
        model = simpleRNN(input_dim = len(x_train[0][0]), output_dim = self.cfg_m.data.dim_out, bidirectional=True, name=self.cfg_proj.model_name) 
        optimizer = torch.optim.AdamW(model.parameters(), lr = self.cfg_m.training.lr_init)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, int(self.cfg_m.training.epochs*len(dataloader_train)))   #very useful
        criterion = self.cross_entropy_regs
        model, _ = self.to_parallel_model(model)
        
        # Train classifier
        model, loss_train_trace = self.basic_train(model, dataloader_train, criterion, optimizer, lr_scheduler)
        
        # Save Weight
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
    
    def cross_entropy_regs(self, model, Yhat, Y, l2_lambda, l1_lambda):    #pred, train_Y
        Y_t = Y
        if Y.dim() == 1:
            Y_t = torch.zeros((Y.shape[0], 2)).to(Y)
            Y_t[:, 1] = Y.data
            Y_t[:, 0] = 1 - Y.data
        loss_mean = F.cross_entropy(Yhat, Y_t)

        if l2_lambda is not None:
            l2_reg = torch.tensor(0., requires_grad=True)
            for name, param in model.named_parameters():
                if 'weight' in name:
                    l2_reg = l2_reg + torch.norm(param, 2)
            l2_reg = l2_lambda * l2_reg
            loss_mean += l2_reg
        if l1_lambda is not None:
            l1_reg = torch.tensor(0., requires_grad=True)
            for name, param in model.named_parameters():
                if 'weight' in name:
                    l1_reg = l1_reg + torch.norm(param, 1)
            l1_reg = l1_lambda * l1_reg
            loss_mean += l1_reg
        return loss_mean
    
    def basic_train(self, model, dataloader_train, criterion, optimizer, lr_scheduler):
        loss_train_trace = []
        for epoch in range(self.cfg_m.training.epochs):
            model.train()
            loss_epoch = []
            for train_X, train_Y, _ , idx in dataloader_train:   

                train_X, train_Y = train_X.float().to(self.device), train_Y.to(self.device)

                
                Y_hat = model(train_X)
                
                Y_hat = Y_hat if torch.is_tensor(Y_hat) else Y_hat[1]
            

                loss = criterion(model, Y_hat, train_Y, l2_lambda = self.cfg_m.l2_lambda, l1_lambda = self.cfg_m.l1_lambda)
                loss_epoch.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                loss_train_trace.append(np.mean(loss_epoch))
        return model, loss_train_trace
    