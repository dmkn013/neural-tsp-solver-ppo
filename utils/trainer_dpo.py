
import os
import time
import copy
import json

import glob
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


from utils.problems.problem_tsp import TSPDataset


class DPOTrainer():
    def __init__(self, model_policy, model_reward, optimizer_policy, optimizer_reward, val_loader, cfg):
        self.policy = model_policy
        self.reward = model_reward
        self.n_nodes = cfg.n_nodes
        self.epoch_size = cfg.epoch_size
        self.batch_size = cfg.batch_size
        self.save_dir = os.path.join(cfg.save_dir, f'tsp_{cfg.n_nodes}', cfg.run_name)
        self.beta = 1.
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        if not os.path.exists(os.path.join(self.save_dir, 'reward')):
            os.makedirs(os.path.join(self.save_dir, 'reward'))
        cfg_dict = vars(cfg)
        with open(os.path.join(self.save_dir, 'args.json'), 'w') as json_file:
            json.dump(cfg_dict, json_file, indent=4)

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
            model_policy.load_state_dict(checkpoint['model'])
            
        self.resume = cfg.resume
        self.mse_loss = nn.MSELoss()
        self.n_rollout = cfg.n_rollout
        self.epoch_start = self.epoch_last+1 if self.resume else 0
        self.policy_old = copy.deepcopy(model_policy)
        self.val_loader = val_loader
        self.optim_policy = optimizer_policy
        self.optim_reward = optimizer_reward
        self.cfg = cfg
        self.log_csv_policy = os.path.join(self.save_dir, 'log.csv')
        self.log_csv_reward = os.path.join(self.save_dir, 'reward', 'log.csv')
        self.mse = nn.MSELoss()


    def fit(self, n_epochs1, n_epochs2, n_epochs3):
        print('--------------------------------------------------------------------------\ntrain policy model\n\n')
        self.train_only_policy(n_epochs1)
        print('--------------------------------------------------------------------------\ntrain reward model\n\n')
        self.train_reward(n_epochs2)
        print('--------------------------------------------------------------------------\ntrain policy model directly')
        self.train_policy_directly(n_epochs3, n_epochs1)


        self.train_reward(n_epochs2)


    def train_only_policy(self, n_epochs):
        loss_val_min = 1e10
        path2model = None
        if not self.resume:
            with open(self.log_csv_policy, 'w') as f:
                f.write('epoch,cost_train,cost_val,time_train,time_val,loss_reward\n')

        for i_epoch in range(self.epoch_start, self.epoch_start+n_epochs):
            self.epoch = i_epoch
            result_str, loss_val = self.train_epoch_policy(i_epoch)
            if loss_val<loss_val_min:
                loss_val_min = loss_val
                model_to_save = copy.deepcopy(self.policy)
                epoch_best = i_epoch
                filename_model = f'epoch-{epoch_best}.pt'
                if path2model is not None:
                    os.remove(path2model)
                path2model = os.path.join(self.save_dir, filename_model)
                torch.save({'model': model_to_save.state_dict()}, path2model)


            with open(self.log_csv_policy, 'a') as f:
                f.write(f'{result_str}\n')
        
    def train_epoch_policy(self, i_epoch):

        train_dataset = TSPDataset(size=self.n_nodes, num_samples=self.epoch_size)
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True,
                                  num_workers=os.cpu_count())

        start = time.time()
        costs = []
        losses_reward = []
        self.policy.train()
        self.policy.set_decode_type("sampling")
        self.policy_old.train()
        self.policy_old.set_decode_type("sampling")

        for batch in tqdm(train_loader, postfix=f'{i_epoch}-th epoch processing'):
            if torch.cuda.is_available():
                batch = batch.cuda()
            loss, cost, loss_reward = self.train_batch_policy(batch)
            costs.append(cost)
            losses_reward.append(loss_reward)

        training_time = time.time() - start
        cost_train = np.mean(costs)
        loss_reward = np.mean(losses_reward)
    
        self.policy.set_decode_type('greedy')
        self.policy.eval()
        val_result = self.validate()
        result_str = f"{i_epoch},{cost_train},{val_result['cost']},"
        result_str += f"{training_time},{val_result['time']},{loss_reward}"
        return result_str, val_result['cost']

    def train_batch_policy(self, batch):
        cost_true, log_p_total, log_p_all, tour = self.policy(batch, n_rollout=self.n_rollout)

        with torch.no_grad():
            reward_pred = self.reward(batch.repeat(self.n_rollout, 1, 1), log_p_all)
        loss_reward = self.mse_loss(reward_pred, -cost_true)

        cost_true = cost_true.view(self.n_rollout, -1).transpose(0, 1) # (batch, sample)
        log_p_total = log_p_total.view(self.n_rollout, -1).transpose(0, 1) # (batch, sample)
        

        baseline = cost_true.mean(dim=1, keepdims=True)
        loss = ((cost_true-baseline)*log_p_total).mean()

        self.policy_old = copy.deepcopy(self.policy)
        self.optim_policy.zero_grad()
        loss.backward()
        self.optim_policy.step()
        return loss.item(), cost_true.mean().item(), loss_reward.item()



    def train_reward(self, n_epochs):
        loss_train_min = 1e10
        path2model = None
        if not self.resume:
            with open(self.log_csv_reward, 'w') as f:
                f.write('epoch,loss\n')

        for i_epoch in range(self.epoch_start, self.epoch_start+n_epochs):
            self.epoch = i_epoch
            result_str, loss_train = self.train_epoch_reward(i_epoch)
            print(f'reward model training @ epoch {i_epoch}: {loss_train=}')
            if loss_train<loss_train_min:
                loss_train_min = loss_train
                model_to_save = copy.deepcopy(self.reward)
                epoch_best = i_epoch
                filename_model = f'epoch-{epoch_best}.pt'
                if path2model is not None:
                    os.remove(path2model)
                path2model = os.path.join(self.save_dir, 'reward', filename_model)
                torch.save({'model': model_to_save.state_dict()}, path2model)


            with open(self.log_csv_reward, 'a') as f:
                f.write(f'{result_str}\n')

    def train_epoch_reward(self, i_epoch):

        train_dataset = TSPDataset(size=self.n_nodes, num_samples=self.epoch_size)
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True,
                                num_workers=os.cpu_count())

        losses = []
        self.policy.set_decode_type('greedy')
        self.policy.eval()

        for batch in tqdm(train_loader, postfix=f'{i_epoch}-th epoch processing'):
            batch = batch.cuda()
            loss = self.train_batch_reward(batch)
            losses.append(loss)

        loss = np.mean(losses)
        result_str = f"{i_epoch},{loss}"
        return result_str, loss
    
    def train_batch_reward(self, points):
        with torch.no_grad():
            cost_true, log_p_total, log_p_all, tour = self.policy(points)

        reward_pred = self.reward(points, log_p_all)
        loss_reward_model = self.mse_loss(reward_pred, -cost_true)
        self.optim_reward.zero_grad()
        loss_reward_model.backward()
        self.optim_reward.step()
        return loss_reward_model.item()

        
    
    def train_policy_directly(self, n_epochs, epoch_start):
        self.policy.train()
        self.policy.set_decode_type("greedy")
        self.policy_old = copy.deepcopy(self.policy)
        self.policy_old.train()
        self.policy_old.set_decode_type("greedy")
        loss_val_min = 1e10
        path2model = None

        
        for i_epoch in range(epoch_start, epoch_start+n_epochs):
            self.epoch = i_epoch
            result_str, loss_val = self.train_epoch_policy_directly(i_epoch)
            print(f'epoch {i_epoch}: {loss_val=}')

            if loss_val<loss_val_min:
                loss_val_min = loss_val
                model_to_save = copy.deepcopy(self.reward)
                epoch_best = i_epoch
                filename_model = f'epoch-{epoch_best}.pt'
                if path2model is not None:
                    os.remove(path2model)
                path2model = os.path.join(self.save_dir, 'reward', filename_model)
                torch.save({'model': model_to_save.state_dict()}, path2model)


            with open(self.log_csv_policy, 'a') as f:
                f.write(f'{result_str}\n')



    def train_epoch_policy_directly(self, i_epoch):
        
        start = time.time()
        costs = []
        losses_reward = []

        train_dataset = TSPDataset(size=self.n_nodes, num_samples=self.epoch_size)
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True,
                                num_workers=os.cpu_count())

        for batch in tqdm(train_loader, postfix=f'{i_epoch}-th epoch processing'):
            batch = batch.cuda()
            cost, loss_reward = self.train_batch_policy_directly(batch)
            losses_reward.append(loss_reward)
            costs.append(cost)

        training_time = time.time() - start
        cost_train = np.mean(costs)
        loss_reward = np.mean(losses_reward)
    
        self.policy.set_decode_type('greedy')
        self.policy.eval()
        val_result = self.validate()
        result_str = f"{i_epoch},{cost_train},{val_result['cost']},"
        result_str += f"{training_time},{val_result['time']},{loss_reward}"
        return result_str, val_result['cost']
               

    def train_batch_policy_directly(self, batch):
        cost_true, log_p_total, log_p_all, tour = self.policy(batch)
        with torch.no_grad():
            log_p_old = self.policy_old(batch, tour_to_be_evaluated=tour)[1]
        kl_loss = F.kl_div(log_p_total, torch.exp(log_p_old), reduction='batchmean') # (1)
        
        # for param in self.reward.parameters():
        #     param.require_grad = False
#        with torch.no_grad():
        reward_pred = self.reward(batch, log_p_all)
        # for param in self.reward.parameters():
        #     param.require_grad = True
        loss = -reward_pred.mean() # + self.beta*kl_loss
        print(reward_pred[:10])
        self.policy_old = copy.deepcopy(self.policy)
        loss_reward = self.mse_loss(reward_pred, -cost_true)
        self.optim_policy.zero_grad()
        #self.optim_reward.zero_grad()

        #loss_reward.backward()
        loss.backward()
        
        self.optim_policy.step()
        #self.optim_reward.step()

        return cost_true.mean().item(), loss_reward.item()

        
        
    def validate(self):
        costs = []
        start = time.time()

        for i_batch, batch in enumerate(self.val_loader):
            batch = batch.cuda()
            with torch.no_grad():
                cost_true, log_p_total, log_p_all, tour = self.policy(batch)
            costs.append(cost_true.mean().item())
        
        cost_mean = np.mean(costs)
        duration = time.time() - start
        print(f'{self.epoch}-th epoch: {cost_mean=}')

        return {'cost': cost_mean, 'time': duration}
    







