
import math
import sys

import torch
from torch import nn

from utils.nets.graph_encoder import GraphAttentionEncoder
from utils.nets.critic import ValueNet
sys.path.append('../')
from utils.problems.problem_tsp import TSP


class AM(nn.Module):
    def __init__(self, cfg):
        super(AM, self).__init__()

        self.embedding_dim = cfg.embedding_dim
        self.hidden_dim = cfg.hidden_dim
        self.n_encode_layers = cfg.n_layers_encoder
        self.decode_type = None
        self.temp = 1
        self.tanh_clipping = cfg.tanh_clipping
        self.n_heads = cfg.n_heads

        
        # Learned input symbols for first action
        self.W_placeholder = nn.Parameter(torch.Tensor(2 * self.embedding_dim))
        self.W_placeholder.data.uniform_(-1, 1)  # Placeholder should be in range of activations

        self.initial_embedder = nn.Linear(2, self.embedding_dim)

        self.encoder = GraphAttentionEncoder(
            n_heads=self.n_heads,
            embed_dim=self.embedding_dim,
            n_layers=self.n_encode_layers,
            normalization=cfg.normalization
        )

        self.project_node_embeddings = nn.Linear(self.embedding_dim, 3 * self.embedding_dim, bias=False)
        self.project_fixed_context = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.project_step_context = nn.Linear(2*cfg.embedding_dim, self.embedding_dim, bias=False)
        self.project_out = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.critic = ValueNet(cfg)

    def set_decode_type(self, decode_type, temp=None):
        self.decode_type = decode_type
        if temp is not None:  # Do not change temperature if not provided
            self.temp = temp
            
    def global2local(self, tour_to_be_evaluated):
        batch_size, n_nodes = tour_to_be_evaluated.shape
        local = torch.zeros_like(tour_to_be_evaluated)
        
        for i_step in range(n_nodes):
            for i_batch in range(batch_size):
                index_global = tour_to_be_evaluated[i_batch, i_step]
                index_past = tour_to_be_evaluated[i_batch, :i_step]
                smaller = (index_past<index_global).sum()
                index_local = index_global - smaller
                local[i_batch, i_step] = index_local
                
        return local
        
        
    def forward(self, input_original, tour_to_be_evaluated=None):
        '''
        input: (batch, node, 2)
        output: 
            cost: (batch, node)
            log_p_total: (batch, node)
            value: (batch, node)
        '''
        points_embedded = self.initial_embedder(input_original)
        batch_size, n_nodes, _ = input_original.shape
        node_indices = torch.arange(n_nodes).unsqueeze(0).repeat(batch_size, 1) # (batch, node)
        if torch.cuda.is_available():
            node_indices = node_indices.cuda()
        if tour_to_be_evaluated is not None:
            tour_to_be_evaluated = self.global2local(tour_to_be_evaluated)
        
        log_ps = []
        instant_rewards = []
        values = []
        tours = []


        batch_id = torch.arange(batch_size)
        if torch.cuda.is_available():
            batch_id = batch_id.cuda()
        input = input_original
        points_embedded_available = points_embedded

        for i in range(n_nodes):
            state = TSP.make_state(input)

            embeddings, _ = self.encoder(points_embedded_available)
            graph_embedding, key_glimpse, val_glimpse, logit_key = self._precompute(embeddings)

            first_and_last = self._get_parallel_step_context(embeddings, state)
            log_p = self._get_log_p(first_and_last, graph_embedding, key_glimpse, 
                                    val_glimpse, logit_key, state)[0]
            if tour_to_be_evaluated is None:
                selected = self._select_node(log_p.exp()[:, 0, :])
            else:
                selected = tour_to_be_evaluated[:, i]
            mask = torch.zeros(batch_size, n_nodes-i).to(bool) # (batch, node)
            mask[batch_id, selected] = True
            current = input[batch_id, selected]
            if i==0:
                first = current
            log_p_selected = log_p.squeeze(1)[batch_id, selected]
            value = self.critic(embeddings, first_and_last) # (batch)
            
            if i==0:
                instant_reward = torch.zeros_like(value).detach()
            if 0<i:
                instant_reward = -self.calc_distance(current, previous)
                    
            instant_rewards.append(instant_reward)

            values.append(value)
            log_ps.append(log_p_selected)
            selected_original = node_indices[batch_id, selected]
            input = input[~mask.unsqueeze(-1).repeat(1, 1, 2)].view(batch_size, n_nodes-i-1, 2)
            node_indices = node_indices[~mask].view(batch_size, n_nodes-i-1)
            if i<n_nodes-1:
                points_embedded_available = points_embedded_available[~mask.unsqueeze(-1).repeat(1, 1, self.embedding_dim)].view(batch_size, n_nodes-i-1, -1)
            previous = current
            tours.append(selected_original)

            i += 1

        reward_final = -self.calc_distance(first, current)
        log_ps = torch.stack(log_ps, 1)
        instant_rewards = torch.stack(instant_rewards, 1) # (batch, node)
        tours = torch.stack(tours, 1)
        cost, mask = TSP.get_costs(input_original, tours)
        values = torch.stack(values, 1)
        return log_ps, instant_rewards, values, cost, reward_final, tours


    def _calc_log_likelihood(self, log_p, tour):
        log_p = log_p.gather(2, tour.unsqueeze(-1)).squeeze(-1) # (batch*sample, node)
        log_p_total = log_p.sum(1) # (batch*sample)
        return log_p_total


    def calc_distance(self, current, previous):
        dif = current - previous
        dif2 = dif.pow(2)
        distance = dif2.sum(dim=1).sqrt() # (batch)
        return distance



    def _select_node(self, probs):

        if self.decode_type == "greedy":
            _, selected = probs.max(1)

        elif self.decode_type == "sampling":
            selected = probs.multinomial(1).squeeze(1)
        else:
            raise NotImplementedError
        return selected

    def _precompute(self, embeddings):

        # The fixed context projection of the graph embedding is calculated only once for efficiency
        graph_embed = embeddings.mean(1)
        # fixed context = (batch_size, 1, embed_dim) to make broadcastable with parallel timesteps
        graph_embedding = self.project_fixed_context(graph_embed)[:, None, :]

        # The projection of the node embeddings for the attention is calculated once up front
        glimpse_key_fixed, glimpse_val_fixed, logit_key_fixed = \
            self.project_node_embeddings(embeddings[:, None, :, :]).chunk(3, dim=-1)

        # No need to rearrange key for logit as there is a single head
        key_glimpse = self._make_heads(glimpse_key_fixed, 1)
        val_glimpse = self._make_heads(glimpse_val_fixed, 1)
        logit_key = logit_key_fixed.contiguous()
        return graph_embedding, key_glimpse, val_glimpse, logit_key


    def _get_log_p(self, first_and_last, graph_embedding, 
                   key_glimpse, val_glimpse, logit_key, state):
        
        # Compute query = context node embedding
        query = graph_embedding + self.project_step_context(first_and_last)

        # Compute keys and values for the nodes
        glimpse_K, glimpse_V, logit_K = key_glimpse, val_glimpse, logit_key

        # Compute the mask
        mask = state.get_mask()

        # Compute logits (unnormalized log_p)
        log_p, glimpse = self._one_to_many_logits(query, glimpse_K, glimpse_V,
                                                logit_K, mask)

        log_p = torch.log_softmax(log_p / self.temp, dim=-1)
        assert not torch.isnan(log_p).any()

        return log_p, mask

    def _get_parallel_step_context(self, embeddings, state):

        current_node = state.get_current_node()
        batch_size = current_node.size(0)
        
        if state.i.item() == 0:
            return self.W_placeholder[None, None, :].expand(batch_size, 1, 
                                                            self.W_placeholder.size(-1))

        index_first_last = torch.cat((state.first_a, current_node), 1)[:, :, None]
        index_first_last = index_first_last.expand(batch_size, 2, embeddings.size(-1))
        first_last = embeddings.gather(1, index_first_last)
        first_last = first_last.view(batch_size, 1, -1)
        return first_last


    def _one_to_many_logits(self, query, glimpse_K, glimpse_V, logit_K, mask):

        batch_size, num_steps, embed_dim = query.size()
        key_size = val_size = embed_dim // self.n_heads

        glimpse_Q = query.view(batch_size, num_steps, self.n_heads, 1, key_size).permute(2, 0, 1, 3, 4)
        query_attention = torch.matmul(glimpse_Q, glimpse_K.transpose(-2, -1))
        compatibility =  query_attention / math.sqrt(glimpse_Q.size(-1))

        heads = torch.matmul(torch.softmax(compatibility, dim=-1), glimpse_V)

        heads = heads.permute(1, 2, 3, 0, 4).contiguous().view(-1, num_steps, 1, embed_dim)
        glimpse = self.project_out(heads)

        final_Q = glimpse
        logits = torch.matmul(final_Q, logit_K.transpose(-2, -1)).squeeze(-2) / math.sqrt(final_Q.size(-1))

        logits = torch.tanh(logits) * self.tanh_clipping
        logits[mask] = -math.inf

        return logits, glimpse.squeeze(-2)


    def _make_heads(self, v, num_steps=None):
        assert num_steps is None or v.size(1) == 1 or v.size(1) == num_steps

        return (
            v.contiguous().view(v.size(0), v.size(1), v.size(2), self.n_heads, -1)
            .expand(v.size(0), v.size(1) if num_steps is None else num_steps, v.size(2), self.n_heads, -1)
            .permute(3, 0, 1, 2, 4)  # (n_heads, batch_size, num_steps, graph_size, head_dim)
        )
