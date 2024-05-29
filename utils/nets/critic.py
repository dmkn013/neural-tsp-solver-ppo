
import torch
from torch import nn

class ValueNet(nn.Module):
    def __init__(self, cfg):
        super(ValueNet, self).__init__()
        self.mlp1 = nn.Sequential(nn.Linear(cfg.embedding_dim, cfg.embedding_dim*2),
                                  nn.ReLU(), 
                                  nn.Linear(cfg.embedding_dim*2, cfg.embedding_dim))
        self.mlp2 = nn.Sequential(#nn.Linear(cfg.embedding_dim*2, cfg.embedding_dim*2),
                                  #nn.ReLU(), 
                                  nn.Linear(cfg.embedding_dim*2, cfg.embedding_dim),
                                  nn.ReLU(), 
                                  nn.Linear(cfg.embedding_dim, cfg.embedding_dim//2),
                                  nn.ReLU(), 
                                  nn.Linear(cfg.embedding_dim//2, 1))
        
        self.context_head = nn.Sequential(#nn.Linear(cfg.embedding_dim*2, cfg.embedding_dim*2),
                                  #nn.ReLU(), 
                                  nn.Linear(cfg.embedding_dim*2, cfg.embedding_dim))
                                  #nn.ReLU(), 
                                  #nn.Linear(cfg.embedding_dim, cfg.embedding_dim))
        self.mlp_emb = nn.Linear(cfg.embedding_dim, cfg.embedding_dim)


    def forward(self, embeddings, first_last):
        '''
        input: 
            embeddings: (batch, n_nodes_available, emb)
            first_last: (batch, 1, emb*2)
        output: (batch)
        '''
        n_nodes_available = first_last.size(1)
        context = self.context_head(first_last.squeeze()) # (batch, emb)
        emb = self.mlp1(embeddings) # (batch, n_nodes_available, emb)
        emb = emb.mean(dim=1) * n_nodes_available # (batch, emb)
        emb = self.mlp_emb(emb) # (batch, emb)
        h = torch.cat([emb, context], dim=1) # (batch, emb*2)
        h = self.mlp2(h).squeeze() # (batch)
        return h

