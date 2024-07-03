
import torch
from torch import nn
import math


class RewardPredicter(nn.Module):

    def __init__(self, cfg):
        super(RewardPredicter, self).__init__()

        self.embedding_dim = cfg.embedding_dim
        self.hidden_dim = cfg.hidden_dim
        self.n_layers = cfg.n_layers_reward
        self.layers = [CNNBlock(cfg).cuda() for _ in range(self.n_layers)]
        self.embedder = nn.Linear(2, self.embedding_dim)
        self.ff1 = nn.Linear(self.embedding_dim, self.hidden_dim)
        self.ff2 = nn.Linear(self.hidden_dim, self.embedding_dim)
        self.ff3 = nn.Linear(self.embedding_dim, self.embedding_dim//4)
        self.ff4 = nn.Linear(self.embedding_dim//4, self.embedding_dim//16)
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()
        
            
            
        
    
    def forward(self, points, log_p_all):
        '''
        points: (batch, node, 2)
        log_p_all: (batch, step, node) step=node
        '''
        cost_matrix = torch.cdist(points, points) # (batch, node, node)
        p_all = torch.exp(log_p_all)
        image = torch.cat([cost_matrix.unsqueeze(-1), p_all.unsqueeze(-1)], dim=-1) # (batch, node, node, 2)
        
        h = self.embedder(image).transpose(1, 3) # (batch, emb, node, node)
        for i_layer in range(self.n_layers):
            h = self.layers[i_layer](h) # (batch, emb, node, node)
        
        h = h.mean(dim=(2, 3)) # (batch, emb)
        h_skip = h
        h = self.ff1(h) # (batch, hidden)
        h = self.relu(h)
        h = self.ff2(h) # (batch, emb)
        h = h + h_skip
        h = self.ff3(h)
        h = self.relu(h)
        h = self.ff4(h)
        h = h.mean(dim=1) # (batch)
        cost_pred = self.softplus(h) # always positive
        
        return -cost_pred


class CNNBlock(nn.Module):

    def __init__(self, cfg):
        super(CNNBlock, self).__init__()

        self.embedding_dim = cfg.embedding_dim
        self.hidden_dim = cfg.hidden_dim
        kernel_size = 3
        self.conv1 = nn.Conv2d(self.embedding_dim, self.embedding_dim, kernel_size=kernel_size, padding=1)
        self.bn1 = nn.BatchNorm2d(self.embedding_dim)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(self.embedding_dim, self.embedding_dim, kernel_size=kernel_size, padding=1)
        self.bn2 = nn.BatchNorm2d(self.embedding_dim)
        
    def forward(self, input):
        h = self.conv1(input)
        h = self.bn1(h)
        h = self.relu(h)
        h = self.conv2(h)
        h = self.bn2(h)
        output = h + input
        output = self.relu(output)
        return output

