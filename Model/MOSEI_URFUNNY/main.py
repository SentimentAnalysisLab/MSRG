import argparse
from solver import Solver
from config import get_config
from data_loader import get_loader
import torch

parser = argparse.ArgumentParser(description='Sentiment Analysis')  # description
parser.add_argument('--dataset', type=str, default='mosei', choices=['mosei', 'ur_funny'],
                    help='dataset to use (default: mosei)')  # dataset
parser.add_argument('--batch_size', type=int, default=64, metavar='N', help='batch size (default: 24)')  # batch_size
parser.add_argument('--embed_dropout', type=float, default=0.6, help='embedding dropout')  # dropout
parser.add_argument('--model', type=str, default='OurModel', help='name of the model to use')  # model's name
parser.add_argument('-f', default='', type=str)
# Tuning
parser.add_argument('--clip', type=float, default=0.6, help='gradient clip value (default: 0.8)')
parser.add_argument('--lr', type=float, default=2e-3, help='initial learning rate (default: 1e-3)')
parser.add_argument('--optim', type=str, default='Adam', help='optimizer to use (default: Adam)')
parser.add_argument('--num_epochs', type=int, default=40, help='number of epochs (default: 40)')
parser.add_argument('--batch_chunk', type=int, default=1, help='number of chunks per batch (default: 1)')
# Logistics
parser.add_argument('--seed', type=int, default=1111, help='random seed')  # random seed
parser.add_argument('--no_cuda', action='store_true', default=True, help='do not use cuda')  # cuda or not
parser.add_argument('--name', type=str, default='MOSEI-TEST',
                    help='name of the trial (default: "MOSEI-TEST")')  # saved model name
# Architecture
parser.add_argument('--lksize', type=int, default=3, help='Kernel size of language projection CNN')
parser.add_argument('--vksize', type=int, default=3, help='Kernel size of visual projection CNN')
parser.add_argument('--aksize', type=int, default=3, help='Kernel size of accoustic projection CNN')
parser.add_argument('--hidden_size', type=int, default=128,help='The size of hiddens')

args = parser.parse_args()
###################################################################
use_cuda = False
if torch.cuda.is_available():
        print("you have cuda and you are using cuda.")
        torch.cuda.manual_seed_all(args.seed)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        use_cuda = True

torch.manual_seed(args.seed)
# np.random.seed(args.seed)
torch.set_default_tensor_type('torch.FloatTensor')
torch.autograd.set_detect_anomaly(True)
####################################################################
#######              Load the dataset                         ######
####################################################################
print("Start loading the data....")
dataset = str.lower(args.dataset.strip())
batch_size = args.batch_size
train_config = get_config(dataset, mode='train', batch_size=args.batch_size )
valid_config = get_config(dataset, mode='valid', batch_size=args.batch_size)
test_config = get_config(dataset, mode='test', batch_size=args.batch_size)
hyp_params = args
train_loader = get_loader(hyp_params, train_config, shuffle=True)
valid_loader = get_loader(hyp_params, valid_config, shuffle=False)
test_loader = get_loader(hyp_params, test_config, shuffle=False)
print('Finish loading the data....')
####################################################################
######                Hyperparameters                       #######
####################################################################
output_dim_dict = {
    'mosei_senti': 1,
    'ur_funny': 2
}
criterion_dict = {
    'ur_funny': 'CrossEntropyLoss'
}
hyp_params.use_cuda = use_cuda
hyp_params.output_dim = output_dim_dict.get(dataset, 1)
hyp_params.criterion = criterion_dict.get(dataset, 'MSELoss')
hyp_params.dataset = hyp_params.data = dataset
hyp_params.batch_chunk = args.batch_chunk
hyp_params.model = str.upper(args.model.strip())
hyp_params.word2id = train_config.word2id
hyp_params.pretrained_emb = train_config.pretrained_emb
hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v = train_config.lav_dim
hyp_params.h_dim = args.hidden_size



if __name__ == '__main__':
    solver = Solver(hyp_params, train_loader=train_loader, dev_loader=valid_loader, test_loader=test_loader,
                    is_train=True)
    solver.train_and_eval()
    exit()
