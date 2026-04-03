from typing import Callable, Union
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F
import torch.nn.functional as F
import math


def model_selector(model_name, row_size, fine_tuning_freeze=0):
    if model_name == 'transformer_ls_stack':
        return Net_transformer_encoder_LSstack(row_size, fine_tuning_freeze)
    else:
        assert False, "Invalid model name"



class Net_transformer_encoder_LSstack(nn.Module):
    def __init__(self, row_size, fine_tuning_freeze=0):
        super(Net_transformer_encoder_LSstack, self).__init__()
        self.d_model = 64

        self.first_fc = nn.Linear(16, self.d_model)
        encoder = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=8, dim_feedforward=256, dropout=0.1, activation="gelu")
        # encoder = TransformerEncoderLayer_save_attention(d_model=self.d_model, nhead=8, dim_feedforward=256, dropout=0.1, activation="gelu")
        self.transformer_encoder = nn.TransformerEncoder(encoder, num_layers=3)
        self.heur_fc1 = nn.Linear(12, self.d_model)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.5)
        self.batch3 = nn.BatchNorm1d(1024)
        self.heur_batch = nn.BatchNorm1d(self.d_model)
        self.fc1 = nn.Linear((row_size+1) * self.d_model, 1024) # 21 * 2 * 64
        self.fc2 = nn.Linear(1024, 12)

        torch.nn.init.xavier_uniform_(self.first_fc.weight)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.heur_fc1.weight)

        self.model_freezing(fine_tuning_freeze)

    # def get_attention_matrix(self):
    #     attention_matrixs = []
    #     for layer in self.transformer_encoder.layers:
    #         attention_matrixs.append(layer.get_attn_weight())

    #     return attention_matrixs

    def forward(self, x, x1):
        #x = torch.tensor_split(x, (7, ), dim=3)
        #x = x[0]
        x = torch.tensor_split(x, 2, dim=1)
        x = torch.cat((x[0], x[1]), dim=3)
        x = self.first_fc(x)
        x1 = self.heur_fc1(x1)
        x1 = torch.unsqueeze(x1, dim=1)
        x = torch.squeeze(x, dim=1)
        x = torch.cat((x, x1), dim=1)
        x = x.permute(1, 0, 2)
        #x = x * math.sqrt(self.d_model)
        x = self.transformer_encoder(x)
        x = F.gelu(x)
        x = x.permute(1, 0, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.batch3(x)
        x = F.leaky_relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        return x
    
    def model_freezing(self, freeze_num):
        if freeze_num == 0:
            pass
        elif freeze_num == 1:
            self.first_fc.requires_grad_(False)
            self.heur_fc1.requires_grad_(False)
        elif freeze_num == 2:
            self.first_fc.requires_grad_(False)
            self.heur_fc1.requires_grad_(False)
            
            self.transformer_encoder.requires_grad_(False)
            self.dropout1.requires_grad_(False)
        elif freeze_num == 3:
            self.first_fc.requires_grad_(False)
            self.heur_fc1.requires_grad_(False)
            
            self.transformer_encoder.requires_grad_(False)
            self.dropout1.requires_grad_(False)

            self.fc1.requires_grad_(False)
            self.batch3.requires_grad_(False)
            self.dropout2.requires_grad_(False)
