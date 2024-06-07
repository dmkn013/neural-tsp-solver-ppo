
import os
import time
import copy

import glob
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
import torch.optim as optim


class Trainer():
    def __init__(self, model, train_loader, val_loader, optimizer, cfg):
        self.model = model
        self.save_dir = os.path.join(cfg.save_dir, f'tsp_{cfg.n_nodes}', cfg.run_name)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        if cfg.resume:
            checkpoints = glob.glob(os.path.join(self.save_dir, '*.pt'))
            epochs = []
            for checkpoint in checkpoints:
                filename = os.path.basename(checkpoint)
                epoch = filename.replace('.pt', '').replace('epoch-', '')
                epochs.append(int(epoch))
            self.epoch_last = max(epochs)
            path2checkpoint = os.path.join(self.save_dir, f'epoch-{self.epoch_last}.pt')
            checkpoint = torch.load(path2checkpoint)
            model.load_state_dict(checkpoint['model'])
        self.resume = cfg.resume
        self.epoch_start = self.epoch_last+1 if self.resume else 0
        self.model_old = copy.deepcopy(model)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epsilon = .1
        self.coef_value = cfg.coef_value
        self.optimizer = optim.Adam(params=model.parameters(), lr=cfg.lr)
        self.cfg = cfg
        self.log_csv = os.path.join(self.save_dir, 'log.csv')
        self.mse = nn.MSELoss()


    def fit(self, n_epochs):
        loss_val_min = 1e10
        path2model = None
        if not self.resume:
            with open(self.log_csv, 'w') as f:
                f.write('epoch,cost_train,cost_val,time_train,time_val,loss_policy,loss_value\n')

        for i_epoch in range(self.epoch_start, self.epoch_start+n_epochs):
            self.epoch = i_epoch
            result_str, loss_val = self.train_epoch(i_epoch)
            if loss_val<loss_val_min:
                loss_val_min = loss_val
                model_to_save = copy.deepcopy(self.model)
                epoch_best = i_epoch
                filename_model = f'epoch-{epoch_best}.pt'
                if path2model is not None:
                    os.remove(path2model)
                path2model = os.path.join(self.save_dir, filename_model)
                torch.save({'model': model_to_save.state_dict()}, path2model)

            with open(self.log_csv, 'a') as f:
                f.write(f'{result_str}\n')
        


    def train_epoch(self, i_epoch):

        start = time.time()
        costs = []
        self.model.train()
        self.model.set_decode_type("sampling")
        self.model_old.train()
        self.model_old.set_decode_type("sampling")
        losses_policy = []
        losses_value = []

        for batch in tqdm(self.train_loader, postfix=f'{i_epoch}-th epoch processing'):
            if torch.cuda.is_available():
                batch = batch.cuda()
            loss_policy, loss_value, cost = self.train_batch(batch)
            losses_policy.append(loss_policy)
            losses_value.append(loss_value)
            costs.append(cost)

        training_time = time.time() - start
        cost_train = np.mean(costs)
    
        self.model.set_decode_type('greedy')
        self.model.eval()
        val_result = self.validate()
        loss_policy_mean = np.mean(losses_policy)
        loss_value_mean = np.mean(losses_value)
        result_str = f"{i_epoch},{cost_train},{val_result['cost']},"
        result_str += f"{training_time},{val_result['time']},{loss_policy_mean},{loss_value_mean}"
        return result_str, val_result['cost']

    def calc_advantage(self, reward, value, reward_final):
        '''
        input:
            reward: (batch, node)
            value: (batch, node)
        output:
            advantage: (batch, node)
        '''
        n_node = reward.size(1)
        value_tgt = torch.zeros_like(reward)
        advantage = torch.zeros_like(reward)
        i_step = 0

        for i_step in range(n_node):
            value_tgt_step = reward[:, i_step:].sum(1) + reward_final
            a = -value[:, i_step] + value_tgt_step
            advantage[:, i_step] = a
            value_tgt[:, i_step] = value_tgt_step

        return advantage, value_tgt


    def train_batch(self, batch):
        log_p, reward, value, cost, reward_final, tour = self.model(batch)
        with torch.no_grad():
            out_old = self.model_old(batch, tour)
        log_p_old, tour_recon = out_old[0], out_old[-1]
        advantage, value_tgt = self.calc_advantage(reward, value, reward_final) # (batch, node)
        advantage = advantage - advantage.mean(dim=1, keepdims=True)
        ratio = torch.exp(log_p-log_p_old) # (batch, node)
        ratio_clipped = torch.clamp(ratio, 1-self.epsilon, 1+self.epsilon)
        loss_policy1 = advantage * ratio
        loss_policy2 = advantage * ratio_clipped
        loss_policy = -torch.min(loss_policy1, loss_policy2).mean()
        mse_value = self.mse(value, value_tgt)
        loss_value = torch.sqrt(mse_value) * self.coef_value
        loss = loss_policy + loss_value
        self.model_old = copy.deepcopy(self.model)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss_policy.item(), loss_value.item(), cost.mean().item()

        
    def validate(self):
        costs = []
        start = time.time()

        for i_batch, batch in enumerate(self.val_loader):
            if torch.cuda.is_available():
                batch = batch.cuda()
            with torch.no_grad():
                log_p, reward, value, cost, reward_final, tour = self.model(batch)
            costs.append(cost.mean().item())

        cost_mean = np.mean(costs)
        duration = time.time() - start
        print(f'{self.epoch}-th epoch: {cost_mean=}')

        return {'cost': cost_mean, 'time': duration}








