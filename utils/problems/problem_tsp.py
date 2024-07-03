from torch.utils.data import Dataset
import torch
import os
import pickle
from .state_tsp import StateTSP


class TSP(object):

    @staticmethod
    def get_costs(dataset, pi):
        # Check that tours are valid, i.e. contain 0 to n -1
        assert (
            torch.arange(pi.size(1), out=pi.data.new()).view(1, -1).expand_as(pi) ==
            pi.data.sort(1)[0]
        ).all(), "Invalid tour"

        d = dataset.gather(1, pi.unsqueeze(-1).expand_as(dataset))

        return (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1) + (d[:, 0] - d[:, -1]).norm(p=2, dim=1)

    @staticmethod
    def make_dataset(*args, **kwargs):
        return TSPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateTSP.initialize(*args, **kwargs)


    
    
class TSPDataset(Dataset):
    
    def __init__(self, filename=None, n_nodes=50, n_instances=1000000):
        super(TSPDataset, self).__init__()

        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.data = [torch.FloatTensor(row) for row in (data[:n_instances])]
        else:
            # Sample points randomly in [0, 1] square
            self.data = [torch.FloatTensor(n_nodes, 2).uniform_(0, 1) for i in range(n_instances)]

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]
