import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"
from model.Network import Actor, Critic
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
from torch import nn
from utils.reward import evaluation
import os
def test(actor,critic,device,evaluator,vaild_set):
    actor =actor.to(device).eval()
    critic =critic.to(device).eval()
    total_cost=0
    cnt =1
    for v in vaild_set:
        xes,p = actor(v)
        xes =xes[:,1:,:]
        critic_est = critic(v)
        reward = evaluator.eval_batch(v.cpu().detach().numpy(),xes.cpu().detach().numpy())
        reward=torch.tensor(reward, dtype=torch.float).cuda()
        r =reward.detach().cpu()
        total_cost += r.mean()
        cnt+=1
    print("average_cost:{0}".format(total_cost/cnt))
def vaild(actor,critic,device,evaluator,model_dir,vaild_set):
    actor =actor.eval()
    critic =critic.eval()
    total_cost=0
    cnt =1
    for v in vaild_set:
        xes,p = actor(v)
        xes =xes[:,1:,:]
        critic_est = critic(v)
        reward = evaluator.eval_batch(batch.cpu().detach().numpy(),xes.cpu().detach().numpy())
        r =reward.detach().cpu()
        total_cost += r.mean()
        cnt+=1
    return total_cost/cnt
def train(bz,n_pin,accumulation,actor,critic,epoch_number,device,max_norma ,max_normc ,evaluator,model_dir,vaild_set,accuary_reward,lr_a=0.002,lr_c=0.002):
    actor_optimer = torch.optim.AdamW(actor.parameters(), lr=lr_a,eps=1e-7)
    critic_optimer = torch.optim.AdamW(critic.parameters(), lr=lr_c,eps=1e-7)
    acheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(actor_optimer , T_max=4, last_epoch=-1)
    ccheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(critic_optimer , T_max=4,last_epoch=-1)
    actor =actor.to(device)
    critic =critic.to(device)
    #batch内平均最优表现
    best_lost = 999999999999;
    #vaildset中最好表现
    best_lost_vaild=best_lost;
    if n_pin<=25:
        steps=120
    elif n_pin>=50 and n_pin<=100:
        steps=80
    else: 
        steps=60
    mse_loss=nn.MSELoss()
    actor.train()
    critic.train()
    for epoch in range(epoch_number):
        step=epoch
        batch = np.random.rand(bz,n_pin, 2)
        xes,p = actor(batch)
        xes =xes[:,1:,:]
        critic_est = critic(batch)
        with torch.no_grad():
            if accuary_reward==1：
                reward = evaluator.eval_batch(batch.cpu().detach().numpy(),xes.cpu().detach().numpy())
            else:
                reward = evaluation(batch,r)
            reward=torch.tensor(reward, dtype=torch.float).cuda()
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
            print(best_lost)
            if losa < best_lost:
                print(losa)
                print("!!!!!!")
                print("!!!!!!")
                print("!!!!!!")
                print("!!!!!!")
                print("!!!!!!")
                best_lost = losa
                torch.save(actor.state_dict(),os.path.join(model_dir,"actor.pth")
                torch.save(critic.state_dict(),os.path.join(model_dir,"critic.pth")
        if step % steps*10 == steps*10-1:
            acheduler.step()
            ccheduler.step()
        if step % 0.1*epoch_number:
           loss=vaild(actor,critic,device,evaluator,vaild_set)
           if loss< best_lost_vaild:
               best_lost_vaild = loss
               torch.save(actor.state_dict(),os.path.join(model_dir,"actor_best.pth")
               torch.save(critic.state_dict(),os.path.join(model_dir,"critic_best.pth")



      