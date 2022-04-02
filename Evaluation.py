import numpy as np
import torch


def evaluation(V, xes):
    n = len(V)    #xes -> torch.long
    batch_size = xes.size(0)
    V = V.repeat(batch_size,1).unsqueeze(0).reshape(batch_size, n, 2)
    edge = np.zeros(batch_size, n, 8, type=np.int32)
    i = 0
    length = 0
    x = V[:,:,0]
    y = V[:,:,1]
    edge[:,:,0] = x
    edge[:,:,2] = edge[:,:,3]= y
    edge[:,:,1] = edge[:,:,4] = edge[:,:,5] = edge[:,:,6] =edge [:,:,7] = x
    v, h, t = torch.unbind(xes,dim=2)#(batch, n , 1)
    for i in range(n-1):
        def sel(src,pt):
            return torch.index_select(src,dim=1,index=pt)
        def setlower(src,tag,pt,if1):
            return torch.min(sel(src,pt),tag)*if1+~if1*sel(src,pt)
        def setupper(src, tag, pt, if1):
            return torch.max(sel(src,pt),tag)*if1+~if1*sel(src,pt)
        vt = v[:, i, 0] #(batch_size,1)
        ht = h[:, i, 1]
        tt = t[:, i, 2]
        Xh = torch.index_select(x,dim=1,index = ht); Xv = torch.index_select(x,dim=1,index=vt)
        Yh = torch.index_select(y,dim=1, index = ht); Yv = torch.index_select(y,dim=1,index= vt)
        yg = torch.abs(Yh-Yv)
        xg = torch.abs(Xh-Xh)
        if1 = ((tt == 0) + ((Xh-Xv) == 0) + ((Yh-Yv) == 0))
        edge[:,:,2].scatter_(dim=1, index=vt,src=torch.min(sel(edge[:,:,2],vt),Yv)*if1+~if1*sel(edge[:,:,2],vt))
        edge[:,:,3].scatter_(dim=1,index=ht,src=torch.max(sel(edge[:,:,3],ht), Yh)*if1+~if1*sel(edge[:,:,3],ht))
        edge[:,:,0].scatter_(dim=1,index=vt,src=torch.min(sel(edge[:,:,0],vt), Xv)*if1+~if1*sel(edge[:,:,0],vt))
        edge[:,:,1].scatter_(dim=1, index=ht,src=torch.max(sel(edge[:,:,1],ht), Xh)*if1+~if1*sel(edge[:,:,1],ht))
        if1 = ~if1
        if2 =(xg > yg)*if1
        edge[:,:,0].scatter_(dim=1,index=ht,src=setlower(edge[:,:0],Xv+yg,ht,if2))  #edge[h][0] = min(edge[h][0], Xv+yg);
        edge[:,:,1].scatter_(dim=1,index=vt,src=setupper(edge[:,:,1],Xv-yg,vt,if2))        # edge[h][1] = max(edge[V][1], Xv-yg)
                if Xv+yg <= Xh:
                    if Yh-Yv >=0:
                        edge[v][5] = max(edge[v][5], Xv+yg)
                    else:
                        edge[v][7] = max(edge[v][7], Xv+yg)
                else:
                    if Yh-Yv >=0:
                        edge[v][6]= min(edge[v][6],Xv-yg)
                    else:
                        edge[v][4]= min(edge[v][4],Xv-yg)
            else:# if2=~if2
                edge[h][2]=min(edge[h][2],); edge[h][3]=max(edge[h][3],)
                if Yv+xg <= Xh:
                    if Xh-Xv >= 0:
                        edge[v][5] = max(edge[v][5],Xh)
                    else:
                        edge[V][6] = min(edge[v][6],Xh)
                else:
                    if Xh-Xv >= 0:
                        edge[v][7] = max(edge[v][7],Xh)
                    else:
                        edge[v][4] = min(edge[v][4],Xh)
    for i in range(n):
        length += (edge[i][1]-edge[i][0])+(edge[i][3]-edge[i][2])+1.414*((edge[i][5]-edge[i][4])+(edge[i][7]-edge[i][6]))
    return length





