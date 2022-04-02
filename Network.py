from torch import tensor
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from Layers import EncoderLayer
from PTM import  PTM
from  EPTM import EPTM
"Encoder 加不加position encoding"
class Encoder(nn.Module):
    ''' A encoder model with s``elf attention mechanism. '''
    def __init__( self,n_layers, n_head, d_k, d_v, d_inner, encoder_d=128,dropout=0.1):
        super(Encoder,self).__init__()
        self.encoder_d = encoder_d
        self.point_emb = nn.Linear(2,encoder_d,bias=False)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(encoder_d, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(encoder_d, eps=1e-6)
    def forward(self,V):
        # -- Forward
        batch_size = V.size(0)
        V = V.reshape(-1,2)
        enc_output = self.point_emb(V.to(torch.float))
        enc_output = enc_output.reshape(batch_size, -1, self.encoder_d)
        enc_output = self.layer_norm(enc_output)
        for enc_layer in self.layer_stack:
            enc_output = enc_layer(enc_output)
        return enc_output
class Actor(nn.Module):
    def __init__(self,hidden_dim,n_layers, n_head, d_k, d_v, d_inner, encoder_d=128,dropout=0.1):
        super(Actor, self,).__init__()
        self.encoder = Encoder(n_layers, n_head, d_k, d_v, d_inner, encoder_d=encoder_d, dropout=dropout)
        self.ePTM = EPTM(encoder_d, hidden_dim)
        self.pTM = PTM(encoder_d, hidden_dim)
        self.xes = Parameter(torch.zeros(1,dtype=torch.long),requires_grad=False)
        self.mask = Parameter(torch.ones(1), requires_grad=False)
    def forward(self,V):
        num_p = V.size(1)
        batch_size = V.size(0)
        mask = self.mask.repeat(num_p).unsqueeze(0).repeat(batch_size, 1).long()     # (batch, seq_len)
        encoder_outputs = self.encoder(V)
        p = torch.ones(batch_size, 1).cuda()
        xes = self.xes.repeat(batch_size*3).reshape(batch_size,3).unsqueeze(1)
        for i in range(num_p-1):
            Ut, p1,mask= self.pTM(xes=xes, mask=mask,encoder_outputs=encoder_outputs)
            Vt, p2 ,mask = self.ePTM(xes=xes, mask=mask,encoder_outputs=encoder_outputs,idx=Ut.squeeze(1))
            p*=p1;p*=p2
            tt =Vt
            s = (Vt >= 2*num_p)  #(batch_size , 1)
            Vt = torch.fmod(Vt, num_p)
            Vt = torch.where(s,Ut,Vt)
            Ut = torch.where(s,tt,Ut)
            Ut = torch.fmod(Ut, num_p)
            t = tt >= num_p
            t = t * (tt < 2 * num_p)
            t = t + (tt >= 3 * num_p)
            xes = torch.cat((xes,(torch.cat((Ut, Vt, t),dim=1)).unsqueeze(1)), dim=1)
        return xes,p
class Critic(nn.Module):
    def __init__(self, hidden_dimc, n_layers, n_head, d_k, d_v, d_inner, encoder_d=128, dropout=0.1):
        super(Critic, self).__init__()
        self.encoder = Encoder(n_layers, n_head, d_k, d_v, d_inner, encoder_d=encoder_d, dropout=dropout)
        self.G =Parameter(torch.FloatTensor(encoder_d), requires_grad=True)
        self.tahn = nn.Tanh()
        self.Relu = nn.ReLU()
        self.glimpse_linear = nn.Linear(encoder_d,hidden_dimc,bias=True)
        self.seven_linear = nn.Linear(hidden_dimc,1,bias=True)
        nn.init.uniform_(self.G, -1, 1)
    def forward(self,V):
        encoded_output = self.encoder(V)
        e = self.tahn(encoded_output)
        G = self.G.unsqueeze(0)
        G = G.repeat(encoded_output.size(0),1)
        e = torch.bmm(e,G.unsqueeze(2))
        e = F.softmax(e,dim=1).squeeze(2)
        glimpse = torch.bmm(e.unsqueeze(1), encoded_output).squeeze(1)
        glimpse = self.Relu(self.glimpse_linear(glimpse)).squeeze(1)
        glimpse = self.seven_linear(glimpse)
        return glimpse
