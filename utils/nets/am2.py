import torch
from torch import nn
import math
from typing import NamedTuple

from .graph_encoder import GraphAttentionEncoder
from utils.problems.problem_tsp import TSP


class AttentionModelFixed(NamedTuple):
    """
    Context for AttentionModel decoder that is fixed during decoding so can be precomputed/cached
    This class allows for efficient indexing of multiple Tensors at once
    """
    node_embeddings: torch.Tensor
    context_node_projected: torch.Tensor
    glimpse_key: torch.Tensor
    glimpse_val: torch.Tensor
    logit_key: torch.Tensor

    def __getitem__(self, key):
        assert torch.is_tensor(key) or isinstance(key, slice)
        return AttentionModelFixed(
            node_embeddings=self.node_embeddings[key],
            context_node_projected=self.context_node_projected[key],
            glimpse_key=self.glimpse_key[:, key],  # dim 0 are the heads
            glimpse_val=self.glimpse_val[:, key],  # dim 0 are the heads
            logit_key=self.logit_key[key]
        )


class AM2(nn.Module):

    def __init__(self, cfg):
        super(AM2, self).__init__()

        self.embedding_dim = cfg.embedding_dim
        self.hidden_dim = cfg.hidden_dim
        self.n_encode_layers = cfg.n_layers_encoder
        self.decode_type = None
        self.temp = 1
        self.tanh_clipping = cfg.tanh_clipping
        self.n_heads = cfg.n_heads

        step_context_dim = 2 * self.embedding_dim  # Embedding of first and last node
        node_dim = 2  # x, y
        
        # Learned input symbols for first action
        self.W_placeholder = nn.Parameter(torch.Tensor(2 * self.embedding_dim))
        self.W_placeholder.data.uniform_(-1, 1)  # Placeholder should be in range of activations

        self.initial_embedder = nn.Linear(node_dim, self.embedding_dim)

        self.encoder = GraphAttentionEncoder(
            n_heads=self.n_heads,
            embed_dim=self.embedding_dim,
            n_layers=self.n_encode_layers,
            normalization=cfg.normalization
        )

        self.project_node_embeddings = nn.Linear(self.embedding_dim, 3 * self.embedding_dim, bias=False)
        self.project_fixed_context = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.project_step_context = nn.Linear(step_context_dim, self.embedding_dim, bias=False)
        assert self.embedding_dim % self.n_heads == 0
        self.project_out = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)


    def set_decode_type(self, decode_type, temp=None):
        self.decode_type = decode_type
        if temp is not None:
            self.temp = temp


    def forward(self, input_original, n_rollout=1):
        '''
        input: (batch, node, 2)
        output: 
            cost: (batch, node)
            log_p_total: (batch, node)
        '''

        points_embedded = self.initial_embedder(input_original)

        embeddings_all, _ = self.encoder(points_embedded)
        embeddings_all = embeddings_all.repeat(n_rollout, 1, 1)
        batch_size, n_nodes, _ = embeddings_all.shape
        batch_id = torch.arange(batch_size).cuda()
        
        log_p_total = []
        tours = []
        node_indices = torch.arange(n_nodes).unsqueeze(0).repeat(batch_size, 1).cuda() # (batch, node)
        node_indices = node_indices[:, 1:]
        tours.append(torch.zeros(batch_size).cuda())
        embed_a = embeddings_all[:, 1:]
        embed_f = embed_l = embeddings_all[:, 0]

        for i in range(n_nodes-1):

            graph_embedding, key_glimpse, val_glimpse, logit_key = self.precompute(embed_a, embed_f, embed_l)
            first_and_last = self.get_parallel_step_context(embed_f, embed_l)
            log_p_all = self.get_log_p(first_and_last, graph_embedding, key_glimpse, val_glimpse, logit_key)
            selected = self.select_node(log_p_all.exp()[:, 0, :])
            mask = torch.zeros(batch_size, n_nodes-i-1).to(bool).cuda() # (batch, node)
            mask[batch_id, selected] = True
            log_p_selected = log_p_all.squeeze(1)[batch_id, selected]
            
            log_p_total.append(log_p_selected)
            selected_original = node_indices[batch_id, selected]
            node_indices = node_indices[~mask].view(batch_size, n_nodes-i-2)
            tours.append(selected_original)
            embed_l = embed_a[batch_id, selected]
            embed_a = embed_a[~mask.unsqueeze(-1).repeat(1, 1, self.embedding_dim)]
            embed_a = embed_a.view(batch_size, n_nodes-i-2, self.embedding_dim)

            i += 1

        log_p_total = torch.stack(log_p_total, 1) # (batch, node-1)
        tours = torch.stack(tours, 1)
        
        input_repeated = input_original.repeat(n_rollout, 1, 1)
        cost = TSP.get_costs(input_repeated, tours.to(torch.int64))
        log_p_total = log_p_total.sum(1)
        
        return cost, log_p_total, tours




    def select_node(self, probs):

        assert (probs == probs).all(), "Probs should not contain any nans"

        if self.decode_type == "greedy":
            _, selected = probs.max(1)

        elif self.decode_type == "sampling":
            selected = probs.multinomial(1).squeeze(1)

        else:
            assert False, "Unknown decode type"
        return selected

    def precompute(self, embed_a, embed_f, embed_l):
        num_steps = 1
        graph_embed = torch.cat([embed_a, embed_f.unsqueeze(1), embed_l.unsqueeze(1)], 1).mean(1)
        graph_embed = self.project_fixed_context(graph_embed)[:, None, :]
        key_glimpse, val_glimpse, logit_key = self.project_node_embeddings(embed_a[:, None, :, :]).chunk(3, dim=-1)
        
        key_glimpse = self.make_heads(key_glimpse, num_steps)
        val_glimpse = self.make_heads(val_glimpse, num_steps)
        logit_key = logit_key.contiguous()
        
        return graph_embed, key_glimpse, val_glimpse, logit_key


    def get_log_p(self, first_and_last, graph_embedding, key_glimpse, val_glimpse, logit_key):
        query = graph_embedding + first_and_last.unsqueeze(1)
        
        log_p = self.one_to_many_logits(query, key_glimpse, val_glimpse, logit_key)

        log_p = torch.log_softmax(log_p / self.temp, dim=-1)

        assert not torch.isnan(log_p).any()

        return log_p

    def get_parallel_step_context(self, embed_f, embed_l):
        """
        Returns the context per step, optionally for multiple steps at once (for efficient evaluation of the model)
        
        :param embeddings: (batch_size, graph_size, embed_dim)
        :param prev_a: (batch_size, num_steps)
        :param first_a: Only used when num_steps = 1, action of first step or None if first step
        :return: (batch_size, num_steps, context_dim)
        """

        first_last = torch.cat([embed_f, embed_l], 1)
        embed_fl = self.project_step_context(first_last)
        return embed_fl



    def one_to_many_logits(self, query, glimpse_K, glimpse_V, logit_K):

        batch_size, num_steps, embed_dim = query.size()
        key_size = val_size = embed_dim // self.n_heads

        # Compute the glimpse, rearrange dimensions so the dimensions are (n_heads, batch_size, num_steps, 1, key_size)
        glimpse_Q = query.view(batch_size, num_steps, self.n_heads, 1, key_size).permute(2, 0, 1, 3, 4)

        # Batch matrix multiplication to compute compatibilities (n_heads, batch_size, num_steps, graph_size)
        compatibility = torch.matmul(glimpse_Q, glimpse_K.transpose(-2, -1)) / math.sqrt(glimpse_Q.size(-1))

        # Batch matrix multiplication to compute heads (n_heads, batch_size, num_steps, val_size)
        heads = torch.matmul(torch.softmax(compatibility, dim=-1), glimpse_V)

        # Project to get glimpse/updated context node embedding (batch_size, num_steps, embedding_dim)
        glimpse = self.project_out(
            heads.permute(1, 2, 3, 0, 4).contiguous().view(-1, num_steps, 1, self.n_heads * val_size))

        final_Q = glimpse
        logits = torch.matmul(final_Q, logit_K.transpose(-2, -1)).squeeze(-2) / math.sqrt(final_Q.size(-1))

        logits = torch.tanh(logits) * self.tanh_clipping

        return logits


    def make_heads(self, v, num_steps=None):
        assert num_steps is None or v.size(1) == 1 or v.size(1) == num_steps

        return (
            v.contiguous().view(v.size(0), v.size(1), v.size(2), self.n_heads, -1)
            .expand(v.size(0), v.size(1) if num_steps is None else num_steps, v.size(2), self.n_heads, -1)
            .permute(3, 0, 1, 2, 4)  # (n_heads, batch_size, num_steps, graph_size, head_dim)
        )
