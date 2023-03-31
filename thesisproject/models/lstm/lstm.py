import torch
from torch import nn

class FixedFeatureLSTM(nn.Module):
    """
    Standard lstm that takes sequences of feature vectors
    """
    def __init__(self, n_features, hidden_size, n_classes, num_layers=1):
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.n_classes = n_classes
        self.num_layers = num_layers
        self.lstm = nn.LSTM(self.n_features, self.hidden_size, self.num_layers, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(self.hidden_size, self.n_classes)

    def forward(self, seq):
        out, _ = self.lstm(seq)
        last_out = out[:, -1, ...] # last output for each sequence in the batch
        out = self.fc(last_out)
        if torch.any(torch.isnan(out)):
            raise Exception("LSTM produces NaN")
        return out


class UnetEncodeLSTM(nn.Module):
    """
    The end-to-end encode-classify network
    It takes a pretrained u-net encoder and a pretrained lstm network, and runs them together, making it possible to train the parameters of both networks at the same time.
    """
    def __init__(self, unet, lstm, hidden_size, n_classes, num_layers=1):
        super().__init__()
        self.n_features = unet.encoding_size
        self.hidden_size = hidden_size
        self.n_classes = n_classes
        self.num_layers = num_layers
        self.unet = unet
        self.lstm = lstm

    def forward(self, slice_batch):
        seq = self.unet.forward(slice_batch, encode=True)
        out = self.lstm(seq)
        return out

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, n_classes, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, n_classes)

    def forward(self, seq):
        out, _ = self.lstm(seq)
        last_out = out[:, -1, ...] # last output for each sequence in the batch
        out = self.fc(last_out)
        return out