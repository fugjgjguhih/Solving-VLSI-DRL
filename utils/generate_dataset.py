import numpy as np
import itertools
import pandas as pd
import argparse
from tqdm import tqdm
import random
seed =22
random.seed(seed)
np.random.seed(seed)
parser = argparse.ArgumentParser()
parser.add_argument('--n_pins', default=15,type=int)
#       生成随机点集
# n_pins 顶点数
# num_b = batch 数目
args = parser.parse_args()
n_pins = args.n_pins;
num_b = 2
batch_size = 256
sums = batch_size*(num_b)
bats =int(batch_size/4)
random_list = list(itertools.product(np.arange(0, 1,0.0001), np.arange(0,1,0.0001)))
t = np.array(random.sample(random_list, n_pins*bats),dtype=np.float32).reshape(-1,n_pins,2)
a = np.array(random.sample(random_list, n_pins),dtype=np.float32).reshape(-1,n_pins,2)
for i in tqdm(range(num_b-1)):
    for ty in range(4):
        for j in range(int(batch_size/4)):
            b = np.array(random.sample(random_list, n_pins*bats),np.float32).reshape(-1,n_pins,2)
            t = np.concatenate((b,t),0).reshape(-1,n_pins,2)
    a = np.concatenate((a,t),0).reshape(-1,n_pins,2)
a = a.reshape(-1,n_pins*2)
df = pd.DataFrame(a)
df.to_csv("./data/dataset1.csv",header=False,index=False, sep=',')



