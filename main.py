from Network import Actor, Critic
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
from reward import evaluation
a =torch.randn(4,3,1)
print(a)
print(a.squeeze(2))
#print(torch.cat((a,a),dim=1))
c =torch.zeros(4).bool()
print(c)

