# model.py
# 這個 block 是要拿來訓練的模型
import torch
from torch import nn
class LSTM_Net(nn.Module):
    def __init__(self, embedding, embedding_dim, hidden_dim, num_layers, dropout=0.5, fix_embedding=True):
        super(LSTM_Net, self).__init__()
        # 製作 embedding layer
        self.embedding = torch.nn.Embedding(embedding.size(0),embedding.size(1))
        self.embedding.weight = torch.nn.Parameter(embedding)
        # 是否將 embedding fix 住，如果 fix_embedding 為 False，在訓練過程中，embedding 也會跟著被訓練
        self.embedding.weight.requires_grad = False if fix_embedding else True
        self.embedding_dim = embedding.size(1)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.classifier = nn.Sequential( nn.Dropout(0.2),
                                         nn.Linear(hidden_dim, 1024),
                                         nn.Dropout(dropout),
                                         nn.PReLU(),
                                         nn.Linear(1024, 1),
                                         nn.Sigmoid() )
    def forward(self, inputs):
        inputs = self.embedding(inputs)
        x, _ = self.lstm(inputs, None)
        # x 的 dimension (batch, seq_len, hidden_size)
        # 取用 LSTM 最後一層的 hidden state
        x = x[:, -1, :] 
        x = self.classifier(x)
        return x

class LSTM_Net_BI(nn.Module):
    def __init__(self, embedding, embedding_dim, hidden_dim, num_layers, dropout=0.5, fix_embedding=True):
        super(LSTM_Net_BI, self).__init__()
        # 製作 embedding layer
        self.embedding = torch.nn.Embedding(embedding.size(0),embedding.size(1))
        self.embedding.weight = torch.nn.Parameter(embedding)
        # 是否將 embedding fix 住，如果 fix_embedding 為 False，在訓練過程中，embedding 也會跟著被訓練
        self.embedding.weight.requires_grad = False if fix_embedding else True
        self.embedding_dim = embedding.size(1)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.classifier = nn.Sequential( nn.Dropout(0.2),
                                         nn.Linear(hidden_dim*2, 1024),
                                         nn.Dropout(dropout),
                                         nn.PReLU(),
                                         nn.Linear(1024, 1),
                                         nn.Sigmoid() )
    def forward(self, inputs):
        inputs = self.embedding(inputs)
        x, _ = self.lstm(inputs, None)
        # x 的 dimension (batch, seq_len, hidden_size)
        # 取用 LSTM 最後一層的 hidden state
        x = x[:, -1, :] 
        x = self.classifier(x)
        return x

class D_Net(nn.Module):
    def __init__(self, embedding_dim, num_layers):
        super(D_Net, self).__init__()
        self.num_layers = num_layers
        self.classifier = nn.Sequential( nn.Linear(embedding_dim, 1024),
                                         nn.Dropout(0.5),
                                         nn.PReLU(),
                                         nn.Linear(1024, 1024),
                                         nn.Dropout(0.5),
                                         nn.PReLU(),
                                         nn.Linear(1024, 512),
                                         nn.Dropout(0.5),
                                         nn.PReLU(),
                                         nn.Linear(512, 128),
                                         nn.Dropout(0.5),
                                         nn.PReLU(),
                                         nn.Linear(128, 1),
                                         nn.Sigmoid() )
    def forward(self, inputs):
        x = self.classifier(inputs.float())
        return x