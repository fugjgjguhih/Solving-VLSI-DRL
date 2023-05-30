import numpy as np
from torch import tensor
import torch
#    reward函数
def evaluation(V, xes):
    n = V.size(1)   #xes -> torch.long
    batch_size = xes.size(0)
    edge = torch.zeros((batch_size, n, 8), dtype=torch.float64).cuda()
    length = 0
    x = V[:,:,0]
    y = V[:,:,1]
    edge[:,:,0] = x
    edge[:,:,2] = edge[:,:,3]= y
    edge[:,:,1] = edge[:,:,4] = edge[:,:,5] = edge[:,:,6] =edge [:,:,7] = x
    v, h, t = torch.unbind(xes,dim=2)#(batch, n , 1)
    for i in range(n-1):
        def sel(src,pt):
            return src.gather(1,pt)
        def setlower(src,tag,pt,if1):
            return torch.min(sel(src,pt),tag)*if1+~if1*sel(src,pt)
        def setupper(src, tag, pt, if1):
            return torch.max(sel(src,pt),tag)*if1+~if1*sel(src,pt)
        vt = v[:, i].unsqueeze(1)
        ht = h[:, i].unsqueeze(1)
        tt = t[:, i].unsqueeze(1)
        Xh = x.gather(1,ht); Xv = x.gather(1,vt)
        Yh = y.gather(1,ht); Yv = y.gather(1,vt)
        yg = torch.abs(Yh-Yv)
        xg = torch.abs(Xh-Xv)
        if1 = ((tt == 0) + ((Xh-Xv) == 0) + ((Yh-Yv) == 0))
        src=torch.min(sel(edge[:,:,2],vt),Yh)*if1+~if1*sel(edge[:,:,2],vt)
        edge[:,:,2].scatter_(dim=1, index=vt,src=torch.min(sel(edge[:,:,2],vt),Yh)*if1+~if1*sel(edge[:,:,2],vt))
        edge[:,:,3].scatter_(dim=1,index=vt,src=torch.max(sel(edge[:,:,3],vt), Yh)*if1+~if1*sel(edge[:,:,3],vt))
        edge[:,:,0].scatter_(dim=1,index=ht,src=torch.min(sel(edge[:,:,0],ht), Xv)*if1+~if1*sel(edge[:,:,0],ht))
        edge[:,:,1].scatter_(dim=1, index=ht,src=torch.max(sel(edge[:,:,1],ht), Xv)*if1+~if1*sel(edge[:,:,1],ht))
        if2 =(xg > yg)*(~if1)
        edge[:,:,0].scatter_(dim=1,index=ht,src=setlower(edge[:,:,0],Xv+yg,ht,if2))  #edge[h][0] = min(edge[h][0], Xv+yg);
        edge[:,:,1].scatter_(dim=1,index=ht,src=setupper(edge[:,:,1],Xv-yg,ht,if2))        # edge[h][1] = max(edge[V][1], Xv-yg)
        if3 = if2*(Xv <= Xh)
        if4 = if3*(Yh >=Yv)
        edge[:,:,5].scatter_(dim =1, index=vt, src=setupper(edge[:,:,5],Xv+yg,vt,if4))
        if4 = if3*(Yh <Yv)
        edge[:,:,7].scatter_(dim = 1, index=vt,src=setupper(edge[:,:,7],Xv+yg,vt,if4))
        if3 = if2*(Xv>Xh)
        if4 = if3*(Yh <Yv)
        edge[:,:,6].scatter_(dim=1, index=vt,src=setlower(edge[:,:,6],Xv-yg,vt,if4))
        if4 = if3 * (Yh >= Yv)
        edge[:, :, 4].scatter_(dim=1, index=vt, src=setlower(edge[:, :, 4], Xv - yg, vt, if4))
        if2 = if1*(xg <= yg)
        edge[:,:,2].scatter_(dim=1,index=ht,src=setlower(edge[:,:,2],Yv+xg,ht,if2))
        edge[:,:,3].scatter_(dim=1,index=ht,src=setupper(edge[:,:,3],Yv-xg,ht,if2))
        if3 = if2*(Yv >= Yh)
        edge[:,:,7].scatter_(dim=1,index=vt,src=setupper(edge[:,:,7],Xh,vt,if3))
        edge[:, :,6].scatter_(dim=1,index =vt ,src =setlower(edge[:,:,6],Xh,vt,if3))
        if3 = if2 * (Yv < Yh)
        edge[:,:,5].scatter_(dim=1,index=vt,src=setupper(edge[:,:,5],Xh,vt,if3))
        edge[:,:,4].scatter_(dim=1,index=vt,src=setlower(edge[:,:,4],Xh,vt,if3))
    length += (torch.sum((edge[:,:,1]-edge[:,:,0]),dim=1)+ torch.sum((edge[:,:,3]-edge[:,:,2]),dim=1))
    length = length.float()
    length += (1.414*torch.sum(edge[:, :, 5]-edge[:, :, 4], dim=1) + 1.414*torch.sum((edge[:, :, 7]-edge[:, :, 6]), dim=1))
    return length





