# coding = utf-8

import torch as t
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class PoetryModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(PoetryModel, self).__init__()
        # hidden_dim 在后面还会用到，所以先存起来
        self.hidden_dim = hidden_dim
        
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        # LSTM初始化参数：
        # input_size：x的特征维度；
        # hidden_size：隐藏层的特征维度；
        # num_layers：lstm隐层的层数，默认为1
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=2)
        self.linear1 = nn.Linear(hidden_dim, vocab_size)
    
    
    def forward(self, input, hidden=None):
        # input放进来前，已经行列互换了，为了并行计算的需要
        seq_len, batch_size = input.size()
        if hidden is None:
            #  h_0 = 0.01*torch.Tensor(2, batch_size, self.hidden_dim).normal_().cuda()
            #  c_0 = 0.01*torch.Tensor(2, batch_size, self.hidden_dim).normal_().cuda()
            h_0 = input.data.new(2, batch_size, self.hidden_dim).fill_(0).float()
            c_0 = input.data.new(2, batch_size, self.hidden_dim).fill_(0).float()
            h_0, c_0 = Variable(h_0), Variable(c_0)
        else:
            h_0, c_0 = hidden
        # size: (seq_len,batch_size,embeding_dim)
        embeds = self.embeddings(input)
        # output size: (seq_len,batch_size,hidden_dim)
        output, hidden = self.lstm(embeds, (h_0, c_0))

        # size: (seq_len*batch_size,vocab_size)
        # 比如说一次训练128个句子，每个句子长度125，那这一批就会预测128 * 125个词
        # 把output压平，每行是一个vector，进行Linear
        output = self.linear1(output.view(seq_len * batch_size, -1))
        # 得到的output的size就是：seq_len * batch_size行，每行有vocab_size列
        return output, hidden