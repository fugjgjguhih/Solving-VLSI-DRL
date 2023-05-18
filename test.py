
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '5,6'
from model.Network import Actor, Critic
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
from utils.Evaluate import Evaluator
from torch import nn
import math
import argparse
seed=22
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
from train.trainer import  test
parser = argparse.ArgumentParser()
parser.add_argument('--action', default='train')
parser.add_argument('--vaild', default="./data/dataset2")
parser.add_argument('--n_pin', default=10,type=int)
parser.add_argument('--model_dir', default='checkpoint')
parser.add_argument('--accuary_reward', default='no')
parser.add_argument('--epochs',default=1000000,type=int)
args = parser.parse_args()

epoch_number = args.epochs
n_pins = args.n_pint
model_dir = args.model_dir
accuary_reward = args.accuary_reward
batch_size = 254
lr_a = 0.000005
lr_c = 0.00004
max_garda = 0.8
max_gardc = 0.8
accumulation = 1
n_pins = 20
print(torch.cuda.is_available())
actor =Actor(hidden_dim=512, n_layers=3, n_head=16 ,d_k=128, d_v=128, d_inner=512)
critic =Critic(hidden_dimc=256, n_layers=3, n_head=16, d_k=128, d_v=128, d_inner=512)
evaluator = Evaluator()
# he initialization
for m in actor.modules():
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        nn.init.kaiming_uniform_(m.weight, mode='fan_in')
        # nn.init.uniform_(m.weight, a=-2, b=2)
for m in critic.modules():
    if isinstance(m, (nn.Conv1d, nn.Linear)):
         nn.init.kaiming_normal_(m.weight,a=math.sqrt(4),mode='fan_in')

vaild_set = pd.read_csv(args.vaild)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
actor.load_state_dict(torch.load(os.path.join(model_dir,"actor.pth")))
critic.load_state_dict(torch.load(os.path.join(model_dir,"critic.pth")))
test(actor, critic,device,evaluator,vaild_set)




