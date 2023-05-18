
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '5,6'
from Network import Actor, Critic
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
from Evaluate import Evaluator
from torch import nn
import pickle
import math
import argparse
from train.trainer import  train
parser = argparse.ArgumentParser()
parser.add_argument('--action', default='train')
parser.add_argument('--dataroot', default="./data/dataset1")
parser.add_argument('--n_pin', default=10,type=int)
parser.add_argument('--model_dir', default='checkpoint')
parser.add_argument('--accuary_loss', default='no')
parser.add_argument('--epochs',default=10000,type=int)
args = parser.parse_args()

epoch_number = args.epochs
n_pins = args.n_pint
criterion = 0
batch_size = 1024
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

if os.path.exists('./model/a23.pth'):
    actor.load_state_dict(torch.load('./model/a23.pth'))
    critic.load_state_dict(torch.load('./model/cc23.pth'))
# V = pd.read_csv('dataset3.csv')
# V = np.array(V, dtype=np.float32)
if os.path.exists('./model/losa23.pkl'):
    with open('./model/losa23.pkl','rb') as f:
        best_a =pickle.load(f)[0]
else:
    best_a =999999999
d=0

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train(batch_size,n_pins,accumulation, actor, critic,epoch_number,device,max_garda,max_gardc,evaluator,best_a,d,lr_a, lr_c)




