import torch.nn as nn
import torch
from torch.nn import Parameter
"batch? 并行化"
"mask = tensor(if vistied = 1 otherwise 0)"
"hidden _dim"
" mask = tf.zeros((self.batch_size, self.max_length)) # mask for actions /"
" encoded_ref = tf.nn.conv1d(actor_encoding, W_ref, 1, "") # actor_encoding is the ref for actions [Batch size, seq_length, n_hidden] ?"
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
        self.context_linear = nn.Conv1d(input_dim, hidden_dim, 1, 1)
        self.V = Parameter(torch.FloatTensor(hidden_dim), requires_grad=True)
        self._inf = Parameter(torch.FloatTensor([float('-inf')]), requires_grad=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=0)

        # Initialize vector V
        nn.init.uniform_(self.V, -1, 1)

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
        # (batch, hidden_dim, seq_len)
        context = context.permute(0, 2, 1)
        ctx = self.context_linear(context)

        # (batch, 1, hidden_dim)
        V = self.V.unsqueeze(0).expand(context.size(0), -1).unsqueeze(1)

        # (batch, seq_len)
        att = torch.bmm(V, self.tanh(inp + ctx)).squeeze(1)
        att = torch.where(mask.bool(),att,self.inf)
        att = 10*self.tanh(att)
        alpha = self.softmax(att)
        return  alpha

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
        self.linear_h = nn.Linear(embedding_dim, hidden_dim,bias=False)
        self.linear_v = nn.Linear(embedding_dim, hidden_dim,bias=False)
        self.linears_h = nn.Linear(embedding_dim, hidden_dim,bias=False)
        self.linears_v = nn.Linear(embedding_dim, hidden_dim,bias=False)
        self.linear_edge = nn.Linear(hidden_dim,hidden_dim,bias=False)
        self.ReLu = nn.ReLU()
        # Used for propagating .cuda() command
        self.runner = Parameter(torch.zeros(1), requires_grad=False)
        self.util =Parameter(torch.zeros(1),requires_grad=False)
    def forward(self,
                encoder_output,
                xes,
                mask
                ):
        """
        Decoder - Forward-pass
        :param Tensor hidden: First decoder's hidden states
        :param Tensor context: Encoder's outputs
        :return: (Output probabilities, Pointers indices), last hidden state
        """

        batch_size = encoder_output.size(0)
        input_length = encoder_output.size(1)
        self.att.init_inf(mask.size())
        def step(h,v,t,el,st):
            if1 = (t == 0).to("cuda:0" if torch.cuda.is_available() else "cpu").unsqueeze(1).expand(batch_size,self.hidden_dim)
            edge = torch.where(if1, self.linear_h(h)+self.linear_v(v), self.linears_v(v)+self.linears_h(h))
            subtree = torch.max(st, self.linear_edge(edge))
            qt = self.ReLu(el + st)
            return qt,edge,subtree
        # Generating arang(input_length), broadcasted across batch_size
        runner = self.runner.repeat(input_length)
        for i in range(input_length):
            runner.data[i] = i
        runner = runner.unsqueeze(0).expand(batch_size, -1).long()
        uity =self.util.repeat(self.hidden_dim)
        uity = uity.unsqueeze(0).expand(batch_size,-1)
        qt = uity; el = uity; st =uity
        if xes.size(1) >= 1:
            for _ in torch.unbind(xes,dim=1):
                h,v,t = torch.unbind(_, 1)
                encoder_h = torch.diagonal(torch.index_select(encoder_output, dim=1, index=h)).permute(1,0).contiguous()
                encoder_v = torch.diagonal(torch.index_select(encoder_output, dim=1, index=v)).permute(1,0).contiguous()
                qt,el,st =step(encoder_h, encoder_v, t, el, st)
        query = qt
        ans = self.att(input=query, context=encoder_output, mask=mask)
     #   ans = torch.where(torch.isnan(ans), torch.full_like(ans, 0), ans) #洗掉ans
        indices = torch.multinomial(ans, 1, replacement=False)  # indices.shape = (batch_size, 1)
        p = torch.gather(ans, 1, index=indices)
        one_hot_pointers = (runner == indices.repeat(1,7)).long()
        # Update mask to ignore seen indices
        mask -=one_hot_pointers
        return indices, p, mask
class PTM(nn.Module):
    """
    Pointer-Net
    """

    def __init__(self,embedding_dim,
                 hidden_dim):
        """
        Initiate Pointer-Net

        :param int embedding_dim: Number of embbeding channels
        :param int hidden_dim: Encoders hidden units
        :param int lstm_layers: Number of layers for LSTMs
        :param float dropout: Float between 0-1
        """

        super(PTM, self).__init__()
        self.decoder = Decoder(embedding_dim, hidden_dim)

    def forward(self, xes,mask, encoder_outputs):
        """
        PointerNet - Forward-pass
        :param Tensor inputs: Input sequence
        :return: Pointers probabilities and indices
        """
        return self.decoder(encoder_outputs,xes,mask)
