import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import pack_padded_sequence


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, dropout=0., act=F.relu):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):
        input = F.dropout(input, self.dropout, self.training)
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        output = self.act(output)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime), attention
        else:
            return h_prime, attention

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
class Lstmen1(torch.nn.Module):
    def __init__(self, hidden_dimension, embedding_dimension):
        super(Lstmen1, self).__init__()
        self.hidden_dim = hidden_dimension
        self.lstm = nn.LSTM(embedding_dimension, hidden_dimension)

    def forward(self, embeds, seq_lens):
        # By default a LSTM requires the batch_size as the second dimension
        # You could also use batch_first=True while declaring the LSTM module, then this permute won't be required
        embeds = embeds.permute(1, 0, 2)  # seq_len * batch_size * embedding_dim

        packed_input = pack_padded_sequence(embeds, seq_lens)
        ah, (hn, cn) = self.lstm(packed_input)
        # two outputs are returned. _ stores all the hidden representation at each time_step
        # (hn, cn) is just for convenience, and is hidden representation and context after the last time_step
        # _ : will be of PackedSequence type, once unpacked, you will get a tensor of size: seq_len x bs x hidden_dim
        # hn : 1 x bs x hidden_dim

        return ah  # bs * hidden_dim


class TextCNN(nn.Module):
    def __init__(self, embedding_size,feature_size,window_sizes):
        super(TextCNN, self).__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=embedding_size, #句子维度
                                    out_channels=feature_size, #输出维度
                                    kernel_size=h,padding=1))#卷积核大小
                          #                              nn.BatchNorm1d(num_features=config.feature_size),
                          # nn.ReLU(),
                          # nn.MaxPool1d(kernel_size=max_doc_len - h + 1)) #句子个数

            for h in window_sizes #卷积核大小
        ])
    def forward(self, x):
        embed_x = x

        # print('embed size 1',embed_x.size())  # 32*35*256
        # batch_size x text_len x embedding_size  -> batch_size x embedding_size x text_len
        embed_x = embed_x.permute(0, 2, 1)
        # print('embed size 2',embed_x.size())  # 32*256*35
        out = [conv(embed_x) for conv in self.convs]  # out[i]:batch_size x feature_size*1
        # for o in out:
        #    print('o',o.size())  # 32*100*1
        out = torch.cat(out, dim=1)  # 对应第二个维度（行）拼接起来，比如说5*2*1,5*3*1的拼接变成5*5*1
        # print(out.size(1)) # 32*400*1
        # out = out.view(-1, out.size(0))
        out=torch.squeeze(out)
        out=torch.transpose(out,1,0)
        # out=out[:-1]
        # print(out.size())  # 32*400
        # print(out)
        return out