import os

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.trainer import Trainer
from utils.nets.am import AM
from utils.problems.problem_tsp import TSPDataset
from config import parse

def run_training():
    cfg = parse()
    torch.manual_seed(cfg.seed)
    model = AM(cfg)
    optimizer = optim.Adam(params=model.parameters(), lr=cfg.lr)

    train_dataset = TSPDataset(size=cfg.n_nodes, num_samples=cfg.epoch_size)
    train_loader = DataLoader(dataset=train_dataset, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=os.cpu_count())

    val_dataset = TSPDataset(size=cfg.n_nodes, num_samples=cfg.n_instances_val)
    val_loader = DataLoader(dataset=val_dataset, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=os.cpu_count())


    trainer = Trainer(model, train_loader, val_loader, optimizer, cfg)
    trainer.fit(cfg.n_epochs)


if __name__=='__main__':
    run_training()

