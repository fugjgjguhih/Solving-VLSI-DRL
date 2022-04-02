import torch.nn as nn
import torch
from torch.nn import Parameter
import torch.nn.functional as F
class Attention(nn.Module):
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

        super(Attention, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.input_linear = nn.Linear(hidden_dim, hidden_dim)
        self.input_linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.input_linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.input_linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.idx_linear = nn.Linear(input_dim,hidden_dim)
        self.context_linear = nn.Conv1d(input_dim, hidden_dim, 1, 1)
        self.context_linears = nn.Conv1d(input_dim, hidden_dim, 1, 1)
        self.context_linearss= nn.Conv1d(input_dim, hidden_dim, 1, 1)
        self.context_linearsss = nn.Conv1d(input_dim, hidden_dim, 1, 1)
        self.V = Parameter(torch.FloatTensor(hidden_dim), requires_grad=True)
        self.V1 = Parameter(torch.FloatTensor(hidden_dim), requires_grad=True)
        self.V2 = Parameter(torch.FloatTensor(hidden_dim), requires_grad=True)
        self.V3 = Parameter(torch.FloatTensor(hidden_dim), requires_grad=True)
        self._inf = Parameter(torch.FloatTensor([float('-inf')]), requires_grad=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=0)

        # Initialize vector V
        nn.init.uniform_(self.V, -1, 1)
        nn.init.uniform_(self.V1, -1, 1)
        nn.init.uniform_(self.V2, -1, 1)
        nn.init.uniform_(self.V3, -1, 1)
    def forward(self, input,
                    context,
                    mask):
            """
            Attention - Forward-pass

            :param Tensor input: Hidden state h
            :param Tensor context: Attention context
            :param ByteTensor mask: Selection mask
            :return: tuple of - (Attentioned hidden state, Alphas)
            """

            # (batch, hidden_dim, seq_len)
            inp = self.input_linear(input).unsqueeze(2).expand(-1, -1, context.size(1))
            inp1 = self.input_linear(input).unsqueeze(2).expand(-1, -1, context.size(1))
            inp2 = self.input_linear(input).unsqueeze(2).expand(-1, -1, context.size(1))
            inp3 = self.input_linear(input).unsqueeze(2).expand(-1, -1, context.size(1))
            # (batch, hidden_dim, seq_len)
            context = context.permute(0, 2, 1)
            ctx = self.context_linear(context)
            ctx1 = self.context_linears(context)
            ctx2 = self.context_linearss(context)
            ctx3 = self.context_linearsss(context)
            # (batch, 1, hidden_dim)
            V = self.V.unsqueeze(0).expand(context.size(0), -1).unsqueeze(1)
            V1 = self.V1.unsqueeze(0).expand(context.size(0), -1).unsqueeze(1)
            V2 = self.V2.unsqueeze(0).expand(context.size(0), -1).unsqueeze(1)
            V3 = self.V3.unsqueeze(0).expand(context.size(0), -1).unsqueeze(1)
            # (batch, seq_len)
            att = torch.bmm(V, self.tanh(inp + ctx)).squeeze(1)
            att1 = torch.bmm(V1, self.tanh(inp1 + ctx1)).squeeze(1)
            att2 = torch.bmm(V2, self.tanh(inp2 + ctx2)).squeeze(1)
            att3 = torch.bmm(V3, self.tanh(inp3 + ctx3)).squeeze(1)
            att = torch.cat((att,att1, att2 ,att3),1)
            att = torch.where(mask.repeat(1,4).bool(), att, self.inf.repeat(1,4))
            att = 10 * self.tanh(att)
            alpha = self.softmax(att)
            return alpha

    def init_inf(self, mask_size):
        self.inf = self._inf.unsqueeze(1).expand(*mask_size)
class Decoder(nn.Module):
    """
    Decoder model for Pointer-Net
    """

    def __init__(self, embedding_dim,
                 hidden_dim):
        """
        Initiate Decoder

        :param int embedding_dim: Number of embeddings in Pointer-Net
        :param int hidden_dim: Number of hidden units for the decoder's RNN
        """

        super(Decoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.att = Attention(embedding_dim, hidden_dim)
        self.index_linear = nn.Linear(embedding_dim,hidden_dim)
        self.linear_h = nn.Linear(embedding_dim, hidden_dim, bias=False)
        self.linear_v = nn.Linear(embedding_dim, hidden_dim, bias=False)
        self.linears_h = nn.Linear(embedding_dim, hidden_dim, bias=False)
        self.linears_v = nn.Linear(embedding_dim, hidden_dim, bias=False)
        self.linear_edge = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # Used for propagating .cuda() command
        self.Relu = nn.ReLU(inplace=True)
        self.runner = Parameter(torch.zeros(1), requires_grad=False)
        self.util =Parameter(torch.zeros(1),requires_grad=False)
    def forward(self,
                encoder_output,
                xes,
                idx,
                mask
                ):
        """
        Decoder - Forward-pass
        :param Tensor hidden: First decoder's hidden states
        :param Tensor context: Encoder's outputs
        :return: (Output probabilities, Pointers indices)torch.diagonal(torch.index_select(encoder_outputs,dim = 1,index=xes[:,0])), last hidden state
        """

        batch_size = encoder_output.size(0)
        input_length = encoder_output.size(1)

        # (batch, seq_len)
        encoder_idx = torch.diagonal(torch.index_select(encoder_output, dim=1, index=idx)).permute(1,0).contiguous()
        self.att.init_inf(mask.size())
        # Generating arang(input_length), broadcasted across batch_size
        runner = self.runner.repeat(input_length)

        def step(h, v, t, el, st):
            if1 = (t == 0).to("cuda:0" if torch.cuda.is_available() else "cpu").unsqueeze(1).expand(batch_size, self.hidden_dim)
            edge = torch.where(if1, self.linear_h(h) + self.linear_v(v), self.linears_v(v) + self.linears_h(h))
            subtree = torch.max(st, self.linear_edge(edge))
            qt = self.Relu(el + st)
            return qt, edge, subtree
        uity = self.util.repeat(self.hidden_dim)
        uity = uity.unsqueeze(0).expand(batch_size, -1).long()
        qt = uity
        el = uity
        st = uity
        if xes.size(1) >= 1:
            for _ in torch.unbind(xes, dim=1):
                h,v,t = torch.unbind(_, 1)
                encoder_h = torch.diagonal(torch.index_select(encoder_output, dim=1, index=h)).permute(1,0).contiguous()
                encoder_v = torch.diagonal(torch.index_select(encoder_output, dim=1, index=v)).permute(1,0).contiguous()
                qt,el,st = step(encoder_h, encoder_v, t, el, st)
        query = qt
        idxs = self.index_linear(encoder_idx)
        query = self.Relu(query+idxs)
        for i in range(input_length):
            runner.data[i] = i
        runner = runner.unsqueeze(0).expand(batch_size, -1).long()
        encoder_idx = self.index_linear(encoder_idx)
        query = self.Relu(query+encoder_idx)
        ans = self.att(input=query, context=encoder_output, mask=mask)
        indices = torch.multinomial(ans, 1, replacement=False)  # indices.shape = (batch_size, 1)
        p = torch.gather(ans, 1, index=indices)
        one_hot_pointers = (runner == indices.repeat(1,7)).long()
        # Update mask to ignore seen indices
        mask -=one_hot_pointers
        return indices, p, mask
class EPTM(nn.Module):
    """
    Pointer-Net
    """

    def __init__(self, embedding_dim,
                 hidden_dim):
        """
        Initiate Pointer-Net
        :param int embedding_dim: Number of embbeding channels
        :param int hidden_dim: Encoders hidden units
        :param int lstm_layers: Number of layers for LSTMs
        :param float dropout: Float between 0-1
        """
        super(EPTM, self).__init__()
        self.decoder = Decoder(embedding_dim, hidden_dim)

    def forward(self,xes,mask,encoder_outputs, idx):
        """
        PointerNet - Forward-pass
        :param Tensor inputs: Input sequence
        :return: Pointers probabilities and indices
        """
        return self.decoder(encoder_output=encoder_outputs, xes=xes, mask=mask, idx=idx)
