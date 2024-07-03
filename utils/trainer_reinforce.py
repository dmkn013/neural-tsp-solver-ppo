
import os
import time
import copy
import json

import glob
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader

from utils.problems.problem_tsp import TSPDataset


class REINFORCETrainer():
    def __init__(self, model, val_loader, optimizer, cfg):
        self.model = model
        self.save_dir = os.path.join(cfg.save_dir, f'tsp_{cfg.n_nodes}', cfg.run_name)

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
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
            model.load_state_dict(checkpoint['model'])

        self.resume = cfg.resume
        self.epoch_start = self.epoch_last+1 if self.resume else 0
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.epoch_size = cfg.epoch_size
        self.batch_size = cfg.batch_size
        self.n_rollout = cfg.n_rollout
        self.cfg = cfg
        self.log_csv = os.path.join(self.save_dir, 'log.csv')


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
            else:
                os.remove(os.path.join(self.save_dir, f'val-epoch-{self.epoch}.png'))


            with open(self.log_csv, 'a') as f:
                f.write(f'{result_str}\n')
        


    def train_epoch(self, i_epoch):
        start = time.time()
        costs = []
        self.model.train()
        self.model.set_decode_type("sampling")
        losses = []
        train_dataset = TSPDataset(n_instances=self.epoch_size)
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=False,
                                  num_workers=os.cpu_count())


        for batch in tqdm(train_loader, postfix=f'{i_epoch}-th epoch processing'):
            loss, cost = self.train_batch(batch)
            losses.append(loss)
            costs.append(cost)

        training_time = time.time() - start
        cost_train = np.mean(costs)
    
        self.model.set_decode_type('greedy')
        self.model.eval()
        val_result = self.validate()
        loss_mean = np.mean(losses)
        result_str = f"{i_epoch},{cost_train},{val_result['cost']},"
        result_str += f"{training_time},{val_result['time']},{loss_mean}"
        return result_str, val_result['cost']

    def train_batch(self, batch):
        batch = batch.cuda()
        batch_size = batch.size(0)
        self.optimizer.zero_grad()

        cost, log_p, tour = self.model(batch, n_rollout=self.n_rollout)

        # cost = cost.view(batch_size, self.n_rollout)
        # log_p = log_p.view(batch_size, self.n_rollout)
        
        # or
        cost = cost.view(self.n_rollout, batch_size).transpose(0, 1)
        log_p = log_p.view(self.n_rollout, batch_size).transpose(0, 1)
        
        baseline = cost.mean(dim=1, keepdims=True)

        loss = ((cost-baseline) * log_p).mean()
        loss.backward()
        self.optimizer.step()
        return loss.item(), cost.mean().item()

        
    def validate(self):
        costs = []
        start = time.time()

        for i_batch, batch in enumerate(self.val_loader):
            batch = batch.cuda()
            with torch.no_grad():
                cost, log_p, tour = self.model(batch)
            self.save_image(batch, tour, i_batch)
            costs.append(cost.mean().item())
        
        cost_mean = np.mean(costs)
        duration = time.time() - start
        print(f'{self.epoch}-th epoch: {cost_mean=}')

        return {'cost': cost_mean, 'time': duration}
    
    def save_image(self, points_batch, tours, i_batch):
        if i_batch!=0:
            return
        points = points_batch[0]
        tour = tours[0]
        sorted = points[tour.to(torch.int64)].cpu().numpy()
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(sorted[:, 0], sorted[:, 1], c='k', marker='o')
        ax.scatter(sorted[0, 0], sorted[0, 1], c='r')
        fig.savefig(os.path.join(self.save_dir, f'val-epoch-{self.epoch}.png'))
#        plt.close()









