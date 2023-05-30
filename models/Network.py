from torch import tensor
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from models.PTM import  PTM
from models.EPTM import  EPTM
import math

class Encoder(nn.Module):
    ''' A encoder model with s``elf attention mechanism. '''
    def __init__( self,n_layers, n_head, encoder_d=128,dropout=0.3):
        super(Encoder,self).__init__()
        self.encoder_d = encoder_d
        self.point_emb = nn.Linear(2,encoder_d,bias=False)
        self.dropout = nn.Dropout(p=dropout)
        self.transformer_encoder =nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=encoder_d, nhead=n_head),
                                                         num_layers=n_layers)
        self.layer_norm = nn.LayerNorm(encoder_d)
    def forward(self,V):
        # -- Forward
        batch_size = V.size(0)
        enc_output = self.point_emb(V.to(torch.float))
        enc_output = enc_output.reshape(batch_size, -1, self.encoder_d)
        enc_output =self.layer_norm(enc_output)
        enc_output = self.transformer_encoder(enc_output)
        return enc_output
class Actor(nn.Module):
    def __init__(self,hidden_dim,n_layers, n_head,  encoder_d=128,dropout=0.3):
        super(Actor, self,).__init__()
        self.hidden_dim = hidden_dim
        self.encoder_d = encoder_d
        self.d_model =encoder_d
        self.d_query = hidden_dim
        self.dropout = dropout

        self.encoder = Encoder(n_layers, n_head, encoder_d=self.encoder_d, dropout=dropout)
        self.ePTM = EPTM(self.encoder_d, self.hidden_dim)
        self.pTM = PTM(self.encoder_d, self.hidden_dim)
        self.xes = Parameter(torch.zeros(1,dtype=torch.long),requires_grad=False)
        self.mask = Parameter(torch.ones(1), requires_grad=False)
        self.zero = torch.zeros(1,dtype=torch.long,requires_grad=False)
        self.q_l1 = nn.Linear(self.d_model, self.d_query, bias=False)
        self.q_l2 = nn.Linear(self.d_model, self.d_query, bias=False)
        self.q_l3 = nn.Linear(self.d_model, self.d_query, bias=False)
        self.q_lx = nn.Linear(self.d_model, self.d_query, bias=False)
        self.q_ly = nn.Linear(self.d_model, self.d_query, bias=False)
        self.q_l1x = nn.Linear(self.d_model, self.d_query, bias=False)
        self.q_l2x = nn.Linear(self.d_model, self.d_query, bias=False)
        self.q_l3x = nn.Linear(self.d_model, self.d_query, bias=False)
        self.q_lxx = nn.Linear(self.d_model, self.d_query, bias=False)
        self.q_lyx = nn.Linear(self.d_model, self.d_query, bias=False)
        self.ctx_linear = nn.Linear(self.d_query, self.d_query, bias=False)
        self.relu=nn.ReLU()
        self.Leakrelu=nn.LeakyReLU()
        self.train()
    def forward(self,V):
        num_p = V.size(1)
        self.n_pin = num_p
        self.degree=num_p
        batch_size = V.size(0)
        mask = self.mask.repeat(num_p).unsqueeze(0).repeat(batch_size, 1).long()     # (batch, seq_len)
        self.ePTM.init_inf(batch_size,num_p)
        self.pTM.init_inf(batch_size,num_p)
        encoder_outputs = self.encoder(V)
        p = torch.ones(batch_size, 1).cuda()
        xes = self.xes.repeat(batch_size*3).reshape(batch_size,3).unsqueeze(1)
        q0 = torch.zeros(batch_size,self.d_query).cuda()
        Ut, p1, mask = self.pTM(input=q0, mask=mask, context=encoder_outputs)
        p*=p1
        q1 = encoder_outputs[torch.arange(batch_size),Ut]
        q2 = q1
        qx = q1
        qy = q1
        mask =mask.long()
        zero =self.zero.repeat(batch_size,num_p)
        self.qzero = self.zero.repeat(batch_size, self.d_query).float().cuda()
        self.zero=zero.cuda()
        context = torch.zeros([batch_size, self.d_query]).cuda()
        mask.scatter_(dim=1, index=Ut.unsqueeze(1), src=self.zero)
        residual = self.q_l1(q1) + self.q_l2(q2) + self.q_lx(qx) + self.q_ly(qy)
        for i in range(num_p-1):
            context = torch.max(context, self.ctx_linear(self.relu(residual)))
            first_q = residual + context
            Ut, p1,mask = self.pTM(input=first_q, mask=mask,context=encoder_outputs)
            q3 = encoder_outputs[torch.arange(encoder_outputs.size(0)), Ut]
            second_query = self.Leakrelu(first_q + self.q_l3(q3))
            Vt, p2, mask = self.ePTM(input=second_query, mask=mask, context=encoder_outputs)
            p*=p1;p*=p2
            tt =Vt.squeeze(1)
            mask=mask.long()
            mask.scatter_(dim=1,index=Ut.unsqueeze(1),src=self.zero)
            Vt = torch.fmod(Vt, num_p).squeeze(1)
            qx = encoder_outputs[torch.arange(encoder_outputs.size(0)), Vt]
            qy = encoder_outputs[torch.arange(encoder_outputs.size(0)), Ut]
            mask.scatter_(dim=1,index=Vt.unsqueeze(1),src=self.zero)
            q1 = q3
            q2 = encoder_outputs[torch.arange(encoder_outputs.size(0)), Vt]
            t = (tt >= num_p) * (tt < 2 * num_p)
            t += (tt >= 3 * num_p)
            Vt = torch.where(t,Ut,Vt)
            Ut = torch.where(t,tt,Ut)
            Ut = torch.fmod(Ut, num_p)
            s = (tt >= 2*num_p)
            residual1 = self.q_l1(q1) + self.q_l2(q2) + self.q_lx(qx) + self.q_ly(qy)
            residual2= self.q_l1x(q1) + self.q_l2x(q2) + self.q_lxx(qx) + self.q_lyx(qy)
            residual=torch.where((~s).unsqueeze(1).repeat(1,self.d_query),self.qzero,residual2)
            residual=torch.where(s.unsqueeze(1).repeat(1,self.d_query),residual,residual1)
            xes = torch.cat((xes,(torch.cat((Ut.unsqueeze(1), Vt.unsqueeze(1), s.unsqueeze(1)),dim=1)).unsqueeze(1)), dim=1)
        self.zero =torch.zeros(1,dtype=torch.long)
        self.pTM.zero = torch.zeros(1,dtype=torch.long)
        return xes,p
class Glimpse(nn.Module):
    def __init__(self, d_model, d_unit):
        super(Glimpse, self).__init__()
        self.tanh = nn.Tanh()
        self.conv1d = nn.Conv1d(d_model, d_unit, 1)
        self.v = nn.Parameter(torch.FloatTensor(d_unit), requires_grad=True)
        self.v.data.uniform_(-(1. / math.sqrt(d_unit)), 1. / math.sqrt(d_unit))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, encs):
        encoded = self.conv1d(encs.permute(0, 2, 1)).permute(0, 2, 1)
        scores = torch.sum(self.v * self.tanh(encoded), -1)
        attention = self.softmax(scores)
        glimpse = attention.unsqueeze(-1) * encs
        glimpse = torch.sum(glimpse, 1)
        return glimpse
class Critic(nn.Module):
    def __init__(self, hidden_dimc, n_layers, n_head, encoder_d=128, dropout=0.1):
        super(Critic, self).__init__()
        self.d_model=encoder_d
        self.d_unit=hidden_dimc
        self.encoder = Encoder(n_layers, n_head, encoder_d=self.d_model, dropout=dropout)
        self.glimpse = Glimpse(self.d_model, self.d_unit)
        self.critic_l1 = nn.Linear(self.d_model, self.d_unit)
        self.critic_l2 = nn.Linear(self.d_unit, 1)
        self.relu = nn.LeakyReLU()
        self.relus = nn.ReLU()
        self.train()
    def forward(self, inputs, deterministic=False):
        critic_encode = self.encoder(inputs)
        glimpse = self.glimpse(critic_encode)
        critic_inner = self.relu(self.critic_l1(glimpse))
        predictions = self.relus(self.critic_l2(critic_inner)).squeeze(-1)

        return predictions

