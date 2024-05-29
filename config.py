
import argparse

def parse():
  parser = argparse.ArgumentParser()


  # train
  parser.add_argument('--seed', type=int, default=1)
  parser.add_argument('--batch_size', type=int, default=512)
  parser.add_argument('--epoch_size', type=int, default=512)
  parser.add_argument('--n_epochs', type=int, default=1000)
  parser.add_argument('--lr', default=1.e-4, type=float)
  parser.add_argument('--run_name', default='test')
  parser.add_argument('--save_dir', default='checkpoints')
  parser.add_argument('--resume', action='store_true')

  # data
  parser.add_argument('--n_nodes', default=20, type=int)
  parser.add_argument('--dataset_path', default='data')
  parser.add_argument('--n_instances_val', type=int, default=512)
  parser.add_argument('--basepath', default='/home/aist/work/am/')
  parser.add_argument('--epsilon', default=.1, type=float)
  parser.add_argument('--coef_value', default=.1, type=float)

  # model
  parser.add_argument('--n_heads', default=8, type=int)
  parser.add_argument('--hidden_dim', default=512, type=int)
  parser.add_argument('--embedding_dim', default=128, type=int)
  parser.add_argument('--n_layers_encoder', type=int, default=2)
  parser.add_argument('--n_layers_decoder', default=2, type=int)
  parser.add_argument('--dropout', default=0.1, type=float)
  parser.add_argument('--normalization', default='batch')
  parser.add_argument('--tanh_clipping', default=10, type=float)


  args = parser.parse_args()
  return args

