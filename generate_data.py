import os
import pickle
import numpy as np

def generate_data():
    n_nodes = 100
    n_instances = 10_000
    data = np.random.uniform(0, 1, (n_instances, n_nodes, 2))
    filename = '20240605.pkl'
    path2file = os.path.join('dataset', f'tsp_{n_nodes}', filename)
    with open(path2file, 'wb') as file:
        pickle.dump(data, file)

    
if __name__=='__main__':
    generate_data()