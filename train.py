
import os

from models.Network import Actor, Critic
import torch
from tqdm import tqdm
import numpy as np
from utils.Evaluate import Evaluator
from torch import nn
import pandas as pd
import math
import argparse
from torch.utils.data import Dataset, DataLoader, TensorDataset
seed=22
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
from trainer.trainer import  train
parser = argparse.ArgumentParser()
parser.add_argument('--vaild', default="./data/dataset1.csv")
parser.add_argument('--n_pin', default=15,type=int)
parser.add_argument('--model_dir', default='checkpoint')
parser.add_argument('--accuary_reward', default=1,type=int)
parser.add_argument('--epochs',default=100000,type=int)
args = parser.parse_args()

epoch_number = args.epochs
n_pins = args.n_pin
model_dir = args.model_dir
#训练时采用精确线长计算还是边界差近似算法：
accuary_reward = args.accuary_reward
batch_size = 128
lr_a = 0.000005
lr_c = 0.00004
max_garda = 0.8
max_gardc = 0.8
accumulation = 1
n_pins = 15
print(torch.cuda.is_available())
actor =Actor(hidden_dim=128, n_layers=8, n_head=8)
critic =Critic(hidden_dimc=128, n_layers=3, n_head=8)
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
vaild_set = torch.tensor(vaild_set.values)
vaild_set = vaild_set.reshape(-1,n_pins,2)
data_set =  TensorDataset(vaild_set,vaild_set)
data_set =DataLoader(dataset=data_set,batch_size=batch_size,shuffle=True,drop_last=False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train(batch_size,n_pins,accumulation, actor, critic,epoch_number,device,max_garda,max_gardc,evaluator,model_dir,vaild_set,accuary_reward,lr_a, lr_c)




