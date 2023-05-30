import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from torch.nn import Parameter

"hidden _dim"
" mask = tf.zeros((self.batch_size, self.max_length)) # mask for actions /"
" encoded_ref = tf.nn.conv1d(actor_encoding, W_ref, 1, "") # actor_encoding is the ref for actions [Batch size, seq_length, n_hidden] ?"
class PTM(nn.Module):
    """
    Attention model for Pointer-Net
    """

    def __init__(self, input_dim,
                 hidden_dim):
        """
        Initiate Attention

        :param int input_dim: Input's diamention
        :param int hidden_dim: Number of hidden units in the attention
        """

        super(PTM, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.input_linear = nn.Linear(hidden_dim, hidden_dim)
        self.context_linear = nn.Conv1d(input_dim, hidden_dim, 1)
        self._V = Parameter(torch.FloatTensor(hidden_dim), requires_grad=True)
        self._inf = Parameter(torch.FloatTensor([float('-inf')]), requires_grad=False)
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()
        # Initialize vector V
        self._V.data.uniform_(-(1. / math.sqrt(hidden_dim)), 1. / math.sqrt(hidden_dim))

    def forward(self, input,
                context,
                mask):
        # (batch, hidden_dim, seq_len)
        inp = self.input_linear(input).unsqueeze(2)
        # (batch, hidden_dim, seq_len)
        context = context.permute(0, 2, 1)
        ctx = self.context_linear(context)

        # (batch, 1, hidden_dim)
        # (batch, seq_len)
        att = torch.bmm(self.V,torch.tanh(inp + ctx))
        att = att.squeeze(1)
        att = 10*self.tanh(att)#为什么要么1要么-1
        with torch.no_grad():
            att = torch.where(mask.bool(), att, self.inf)
        ans = self.softmax(att)
        p, indices = torch.max(ans, dim=1)  # indices.shape = (batch_size, 1)
        return indices, p.unsqueeze(1), mask.bool()

    def init_inf(self,bz, mask_size):
        self.inf = self._inf.unsqueeze(1).expand(bz,mask_size)
        self.V =self._V.repeat(bz,1).unsqueeze(1)

