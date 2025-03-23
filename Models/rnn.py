import torch

class simpleRNN(torch.nn.Module):
    def __init__(self, input_dim, output_dim, bidirectional = False, seq2seq = False, adversarial = False, name = "GRU"):
        super(simpleRNN, self).__init__()
        if adversarial:
            self.hidden_size = 32
            self.num_layers = 2
        else:
            self.hidden_size = 32
            self.num_layers = 3
            
        self.bidirectional = bidirectional
        self.seq2seq = seq2seq
        self.name = name

        if self.name == "GRU":
            self.rnn = torch.nn.GRU(input_dim, self.hidden_size, self.num_layers, batch_first=True, bidirectional = self.bidirectional)
        elif self.name == "RNN":
            self.rnn = torch.nn.RNN(input_dim, self.hidden_size , self.num_layers, batch_first=True, bidirectional = self.bidirectional)
        elif self.name == "LSTM":
            self.rnn = torch.nn.LSTM(input_dim, self.hidden_size, self.num_layers, batch_first=True, bidirectional = self.bidirectional)

        if self.bidirectional:
            self.fc = torch.nn.Linear(self.hidden_size * 2, output_dim)
        else:
            self.fc = torch.nn.Linear(self.hidden_size, output_dim)

    def forward(self, x, hidden_state = False, id = "0"):
        # Initialize hidden state with zeros
        if self.bidirectional:
            h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        else:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate RNN
        if self.name != "LSTM":
            out, _ = self.rnn(x, h0)
        else:
            out, _ = self.rnn(x, (h0, c0))
            
        if self.seq2seq:
            return self.fc(out)
        
        if id == "3":
            return out[:, -1, :]
        
        # Decode the hidden state of the last time step
        final = self.fc(out[:, -1, :])
        
        if hidden_state:
            return final, out

        return final
        
class RNN_temporal_harmonization(torch.nn.Module):
    # Proposed method
    feature_idx = None
    def __init__(self, input_dim, sbj_dim, task_in_dim, task_out_dim, adversarial = False, name = "GRU"):
        super(RNN_temporal_harmonization, self).__init__()
        self.intput_dim = input_dim
        self.feature_mapping = simpleRNN(input_dim, task_in_dim, bidirectional=True, seq2seq=True, name=name)
        self.out_sbj = simpleRNN(task_in_dim, sbj_dim, bidirectional=True, adversarial=adversarial, name=name)
        self.out_task = simpleRNN(task_in_dim, task_out_dim, bidirectional=True, name=name)
        
    def forward(self, x, id, verbose = False):
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
            return self.out_task(feature)
        elif id == "3":
            # Get last hidden state
            return self.out_task(feature, id = id)