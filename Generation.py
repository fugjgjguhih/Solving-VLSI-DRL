import numpy as np
import itertools
import pandas as pd
from tqdm import tqdm
import random
n_pins = 7
num_b = 9
batch_size = 64
sums = batch_size*(num_b)
random_list = list(itertools.product(range(0, 30), range(0, 30)))
a = np.zeros((batch_size,2),dtype=np.int32)
t = np.array(random.sample(random_list, n_pins), dtype=np.int32)
for i in tqdm(range(num_b-1)):
    for t in range(4):
        for j in range(int(batch_size/4)):
            b = np.array(random.sample(random_list, n_pins), dtype=np.int32)
            t = np.concatenate((b,t))
    a = np.concatenate((a,t))
a = a.reshape(sums,n_pins,2)
print(a.reshape(sums,n_pins,2))
a = a.reshape(sums,-1)
df = pd.DataFrame(a)
df.to_csv("dataset.csv", index=False, sep=',')
V = pd.read_csv('dataset.csv')
V =np.array(V,dtype=np.int32)
print(V.reshape(sums, n_pins, 2))

