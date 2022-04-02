from Network import Actor, Critic
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
from reward import evaluation
def train(dataloader,actor,critic,epoch_number,device,lr_a=0.001,lr_c=0.001):
    actor_optimer = torch.optim.Adam(actor.parameters(), lr=lr_a)
    critic_optimer = torch.optim.Adam(critic.parameters(), lr=lr_c)
    actor =actor.to(device)
    critic =critic.to(device)
    for epoch in range(epoch_number):
        actor.train()
        critic.train()
        for step,batch in tqdm(enumerate(dataloader)):
            batch,_ =batch
            batch =batch.reshape(batch_size,-1,2)
            batch = batch.to(device)
            xes,p = actor(batch)
            xes =xes[:,1:,:]
            logp = torch.log(p)
            reward = evaluation(batch,xes)
            critic_est = critic(batch)
            advantage = (reward - critic_est)
            actor_loss = torch.mean(advantage.detach()*logp.sum(dim=1))
            critic_loss = torch.mean(advantage ** 2)
            actor_optimer.zero_grad()
            actor_loss.backward()
            actor_optimer.step()
            #
            critic_optimer.zero_grad()
            critic_loss.backward()
            critic_optimer.step()
        torch.save(critic.state_dict(), 'c.pth')
        torch.save(actor.state_dict(), 'a.pth')

epoch_number = 20
criterion = 0
batch_size = 64
lr_a =0.001
lr_c=0.001
actor =Actor(hidden_dim=360, n_layers=3, n_head=16 ,d_k=16, d_v=16, d_inner=512)
critic =Critic(hidden_dimc=256, n_layers=3, n_head=16, d_k=16, d_v=16, d_inner=512)
V = pd.read_csv('dataset.csv')
V =np.array(V, dtype=np.int32)
V.reshape(-1, 7, 2)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataset =torch.tensor(V)
dataset =torch.utils.data.TensorDataset(dataset,dataset)
dataloader =torch.utils.data.DataLoader(dataset=dataset,batch_size=64,shuffle=True,num_workers=0,drop_last=True)
train(dataloader, actor, critic,epoch_number,device, lr_a, lr_c)



