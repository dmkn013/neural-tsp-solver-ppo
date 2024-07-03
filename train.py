import os

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.trainer_ppo import PPOTrainer
from utils.trainer_dpo import DPOTrainer
from utils.trainer_reinforce import REINFORCETrainer
#from utils.nets.am import AM
from utils.nets.am_original import AM
from utils.nets.am2 import AM2

from utils.problems.problem_tsp import TSPDataset
from config import parse


def run_training():
    
    cfg = parse()
    torch.manual_seed(cfg.seed)
    model = AM2(cfg).cuda()

    filename = '20240605.pkl'
    path2file = os.path.join('dataset', f'tsp_{cfg.n_nodes}', filename)

    val_dataset = TSPDataset(filename=path2file)
    val_loader = DataLoader(dataset=val_dataset, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=os.cpu_count())

    optimizer = optim.Adam(params=model.parameters(), lr=cfg.lr)        
    trainer = REINFORCETrainer(model, val_loader, optimizer, cfg)
    trainer.fit(cfg.n_epochs)


if __name__=='__main__':
    run_training()

