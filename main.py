# coding: utf-8

import sys
import os
from data import get_data
from config import Config
from model import PoetryModel
from torch import nn, Tensor
import torch as t
from torch.autograd import Variable
# 可视化相关
from utils import Visualizer
import tqdm
from torchnet import meter
import ipdb

opt = Config()
device = t.device("cuda:0" if t.cuda.is_available() else "cpu")

if opt.use_gpu and (not t.cuda.is_available()) :
    opt.use_gpu = False

# 训练代码
def train(**kwargs):
    for k, v in kwargs.items():
        setattr(opt, k, v)
    
    vis = Visualizer(env=opt.env)
    
    # 获取数据
    data, word2ix, ix2word = get_data(opt)
    data = t.from_numpy(data)
    dataloader = t.utils.data.DataLoader(data, batch_size=opt.batch_size, shuffle=True, num_workers=2)
    
    # 定义model
    model = PoetryModel(len(word2ix), opt.embedding_dim, opt.hidden_dim)
    # 优化器
    optimizer = t.optim.Adam(model.parameters(), lr=opt.lr)
    # Loss Function
    criterion = nn.CrossEntropyLoss()
    
    
    
    # 使用预训练的模型，为了可持续训练
    if opt.model_path and os.path.exists(opt.model_path):
        model.load_state_dict(t.load(opt.model_path))
    
    # GPU related

    if opt.use_gpu:
        model = model.to(device)
        criterion = criterion.to(device)
    
    # loss 计量器
    loss_meter = meter.AverageValueMeter()
    
    # for loop
    for epoch in range(opt.epoch):
        loss_meter.reset()
        
        # for : batching dataset
        for i, data_ in tqdm.tqdm(enumerate(dataloader)):
            
            # 训练
            # data_ 
            # size: [128, 125]  每次取128行，每行一首诗，长度为125
            # type: Tensor
            # dtype: torch.int32 应该转成long
            
            # 这行代码信息量很大：
            # 第一步：int32 to long
            # 第二步：将行列互换，为了并行计算的需要
            # 第三步：将数据放置在连续内存里，避免后续有些操作报错
            data_ = data_.long().transpose(0, 1).contiguous()
            
            # GPU related
            if opt.use_gpu:
                data_ = data_.to(device)
            
            # 到这里 data_.dtype又变成了torch.int64
            # print(data_.dtype)
            
            # 清空梯度
            optimizer.zero_grad()
            
            # 错位训练，很容易理解
            # 把前n-1行作为input，把后n-1行作为target  :  model的输入
            # 这么做还是为了并行计算的需要
            # input_ 加下划线是为了和built_in function input区分开
            input_, target = data_[:-1, :], data_[1:, :]
            
            # model的返回值 output和hidden
            # 这里hidden没什么用
            output, _ = model(input_)
            
            # 计算loss
            target = target.view(-1)
            
            # 新的target.size() [15872]  124 * 128 = 15872
            # output.size()  [15872, 8293] 8293 是词汇量的大小
            
            loss = criterion(output, target)
            
            # 反向传播
            loss.backward()
            
            # optimizer梯度下降更新参数
            optimizer.step()
            
            loss_meter.add(loss.data[0])

            # 可视化
            if (1 + i) % opt.plot_every == 0:

                if os.path.exists(opt.debug_file):
                    ipdb.set_trace()

                vis.plot('loss', loss_meter.value()[0])

                # 诗歌原文
                poetrys = [[ix2word[_word.item()] for _word in data_[:, _iii]]
                           for _iii in range(data_.size(1))][:16]
                vis.text('</br>'.join([''.join(poetry) for poetry in poetrys]), win=u'origin_poem')

                gen_poetries = []
                # 分别以这几个字作为诗歌的第一个字，生成8首诗
                for word in list(u'春江花月夜凉如水'):
                    gen_poetry = ''.join(generate(model, word, ix2word, word2ix))
                    gen_poetries.append(gen_poetry)
                vis.text('</br>'.join([''.join(poetry) for poetry in gen_poetries]), win=u'gen_poem')
        # 迭代一次epoch，保存一下模型
        t.save(model.state_dict(), '%s_%s.pth' % (opt.model_prefix, epoch))


# 生成诗歌
def generate(model, start_words, ix2word, word2ix, prefix_words=None):
    """
    给定几个词，根据这几个词接着生成一首完整的诗歌
    start_words：u'春江潮水连海平'
    比如start_words 为 春江潮水连海平，可以生成：
    prefix_words: 用来控制意境

    """
    results = list(start_words)
    start_word_len = len(start_words)
    
    # 手动设置第一个词为<START>
    # 之所以把size变成[1, 1]是因为model的输入就是二维的
    input_ = Tensor([word2ix['<START>']]).view(1, 1).long()
    if opt.use_gpu:
        input_ = input_.to(device)
    hidden = None
    
    # 控制意境
    if prefix_words:
        for word in prefix_words:
            output, hidden = model(input_, hidden)
            input_ = input_.data.new([word2ix[word]]).view(1, 1)
    
    # 开始逐词生成诗歌
    for i in range(opt.max_gen_len):
        output, hidden = model(input_, hidden)
        
        if i < start_word_len:
            w = results[i]
            input_ = (input_.data.new([word2ix[w]])).view(1, 1)
        else:
            # 概率最大的那个词的索引
            # output: size [1, 8293] .data size [1, 8293]
            # output.data[0] 取第一行 type: torch.Tensor
            
            # 这行代码信息量很大，首先data是一个二维数组，size [1, 8293]
            # data[0]取第一行
            # .top(1) 一维数组，取出最大的那个，返回值是个tuple (num, index)
            # topk(1)[1][0] 取出index，这个index是个list，取出第一个
            # item() 获取Python数值
            top_index = output.data[0].topk(1)[1][0].item()
            w = ix2word[top_index]
            # 新预测出现的词，加入进去
            results.append(w)
            input_ = (input_.data.new([top_index])).view(1, 1)
        if w == '<EOP>':
            del results[-1]
            break
    return results

def gen_acrostic(model, start_words, ix2word, word2ix, prefix_words=None):
    """
    生成藏头诗
    start_words : u'深度学习'
    生成：
    深木通中岳，青苔半日脂。
    度山分地险，逆浪到南巴。
    学道兵犹毒，当时燕不移。
    习根通古岸，开镜出清羸。
    """
    results = []
    start_word_len = len(start_words)
    
    # 手动设置第一个词为<START>
    # 之所以把size变成[1, 1]是因为model的输入就是二维的
    input_ = Tensor([word2ix['<START>']]).view(1, 1).long()
    if opt.use_gpu:
        input_ = input_.to(device)
    hidden = None
    
    index = 0
    pre_word = '<START>'
    
    # 控制意境
    if prefix_words:
        for word in prefix_words:
            output, hidden = model(input_, hidden)
            input_ = input_.data.new([word2ix[word]]).view(1, 1)
    
    # 开始逐词生成诗歌
    for i in range(opt.max_gen_len):
        output, hidden = model(input_, hidden)
        top_index = output.data[0].topk(1)[1][0].item()
        w = ix2word[top_index]
        
        # 如果遇到这些符号，就把藏头词送进去
        if pre_word in [u'。', u'！', '<START>']:
            if index == start_word_len:
                break
            else:
                w = start_words[index]
                index += 1
                input_ = (input_.data.new([word2ix[w]])).view(1, 1)
        else:
            # 否则的话，就把上次的输出作为输入送进去
            input_ = (input_.data.new([word2ix[w]])).view(1, 1)
        results.append(w)
        pre_word = w
    return results
    

def gen(**kwargs):
    """
    gen 提供命令行接口
    """
    for k, v in kwargs.items():
        setattr(opt, k, v)
    
    data, word2ix, ix2word = get_data(opt)
    model = PoetryModel(len(word2ix), opt.embedding_dim, opt.hidden_dim)
    # todo : what is it???
    map_location = lambda s, l: s
    state_dict = t.load(opt.model_path, map_location=map_location)
    model.load_state_dict(state_dict)
    
    if opt.use_gpu:
        model = model.to(device)
    
    # 默认Python 3.X
    if opt.start_words.isprintable:
        start_words = opt.start_words
        prefix_words = opt.prefix_words
    else:
        # todo : what is it???
        start_words = opt.start_words.encode('ascii', 'surrogateescape').decode('utf8')
        prefix_words = opt.prefix_words.encode('ascii', 'surrogateescape').decode('utf8') if opt.prefix_words else None
    
    # 半角替换成全角
    start_words = start_words.replace(',', u'，').replace('.', u'。').replace('?', u'？')
    
    # 选择合适的生成函数
    gen_poetry = gen_acrostic if opt.acrostic else generate
    result = gen_poetry(model, start_words, ix2word, word2ix, prefix_words)
    print(''.join(result))

if __name__ == '__main__':
    import fire

    fire.Fire()