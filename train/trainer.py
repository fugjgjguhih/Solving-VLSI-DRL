import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"
from Network import Actor, Critic
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
from Evaluate import Evaluator
from torch import nn
import os
import pickle
import argparse
def train(bz,n_pin,accumulation,actor,critic,epoch_number,device,max_norma ,max_normc ,evaluator,ba,d,lr_a=0.002,lr_c=0.002):
    actor_optimer = torch.optim.AdamW(actor.parameters(), lr=lr_a,eps=1e-7)
    critic_optimer = torch.optim.AdamW(critic.parameters(), lr=lr_c,eps=1e-7)
    acheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(actor_optimer , T_max=4, last_epoch=-1)
    ccheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(critic_optimer , T_max=4,last_epoch=-1)
    actor =actor.to(device)
    critic =critic.to(device)
    best_losta = ba #1230
    d=d
    if d==0:
        steps=120
    elif d==1:
        steps=80
    else:
        steps=60
    mse_loss=nn.MSELoss()
    actor.train()
    critic.train()
    for epoch in range(epoch_number):
        step=epoch
        batch = np.random.rand(bz,n_pin, 2)
        # batch,_ =batch
        # batch =batch.reshape(_.size(0),-1,2)
        # batch = batch.to(device)
        xes,p = actor(batch)
        xes =xes[:,1:,:]
        critic_est = critic(batch)
        reward = evaluator.eval_batch(batch.cpu().detach().numpy(),xes.cpu().detach().numpy())
        reward=torch.tensor(reward, dtype=torch.float).cuda()
        with torch.no_grad():
            disadvantage = reward - critic_est
        actor_loss = torch.mean(disadvantage * p)
        critic_loss = mse_loss(critic_est,reward)
        critic_loss = critic_loss
        loss =actor_loss+critic_loss
        actor_optimer.zero_grad()
        critic_optimer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norma)
        torch.nn.utils.clip_grad_norm_(critic.parameters(), max_normc)
        actor_optimer.step()
        critic_optimer.step()
        if step % (steps*1.2) == steps*1.2-1:
            print(reward[0])
            print(critic_est[0])
            print('Epoch ' + str(epoch) + ' : ' + str(step // 200) + ' , LOSSa =' + str(actor_loss*accumulation))
            print('Epoch ' + str(epoch) + ' : ' + str(step // 200) + ' , LOSSc =' + str(critic_loss*accumulation))


        if step % (steps*1.6) == (steps*1.6-1):
            r =reward.detach().cpu()
            losa = r.mean()
            print(critic_est[0])
            print(xes[0])
            print(reward[0])
            print(losa)
            print(best_losta)
            if losa < best_losta:
                print(losa)
                print("!!!!!!")
                print("!!!!!!")
                print("!!!!!!")
                print("!!!!!!")
                print("!!!!!!")
                best_losta = losa
                ls = [losa]
                if d==0:
                    with open("./model/losa23.pkl" ,'wb') as f:
                        pickle.dump(ls,f)
                    torch.save(critic.state_dict(), './model/cc23.pth')
                    torch.save(actor.state_dict(), './model/a23.pth')
                elif d==1:
                    with open("./model/losa20.pkl", 'wb') as f:
                        pickle.dump(ls, f)
                    torch.save(critic.state_dict(), './model/cc20.pth')
                    torch.save(actor.state_dict(), './model/a20.pth')
                else:
                    with open("./model/losa30.pkl", 'wb') as f:
                        pickle.dump(ls, f)
                    torch.save(critic.state_dict(), './model/cc30.pth')
                    torch.save(actor.state_dict(), './model/a30.pth')
        if step % steps*10 == steps*10-1:
            acheduler.step()
            ccheduler.step()
        step+=1;
    ccheduler.step()
    acheduler.step()
