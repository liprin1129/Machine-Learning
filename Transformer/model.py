import torch
import torch.nn as nn
import math

#print(torch.cuda.is_available())

class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):
    "Implement the PE function"
    def __init__(self, d_model: int, dropout: float, max_len:int=5000) -> None:
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        #self.d_model = d_model
        #self.max_len = max_len
        

        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(max_len, d_model)
        # Create a vector of shape (max_len, 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        power_factor = torch.arange(0, max_len, 2) / d_model
