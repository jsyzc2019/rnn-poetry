# AI 诗人


这是陈云的《深度学习框架PyTorch：入门与实践》第九章的AI诗人代码，我自己手敲的，并附带了详细的注解，不放过每个细节。这个小demo是利用RNN自动写诗。

可以看看一首诗：

深山万里外，不见三月风。

度月出门夜，复来无定同。

学道到山中，出门望层穹。

习习一为内，有路不逡长。

这是一首藏头诗，每句第一个词，组合起来就是“深度学习”，是不是很nice。



## 文件组织架构

程序主要包含以下几个功能：

- 根据唐诗数据训练模型
- 给定的诗的开头和意境(指定一个有意境的诗句)，续写后面的诗句
- 给定的诗的开头和意境,写藏头诗

程序文件的组织结构：
```
|---checkpoints/
|---config.py
|---data.py
|---model.py
|---main.py
|---utils.py
|---tang.npz
|---model.ipynb
|---main.ipynb
|---look_data.ipynb
```

其中：

- `checkpoints/`: 用于保存训练好的模型，可使程序在异常退出后仍能重新载入模型，恢复训练
- `config.py`: 模型训练的配置，比如learning rate, epoch, path, batch_size等
- `data.py`: 读取tang.npz的数据并返回data, word2ix, ix2word；look_data.ipynb就是看看这个数据集的细节的，如type，size等，能够帮助你更好的理解数据
- `model.py`: 模型的定义，很简单的，一看就懂，一个embedding，一个LSTM，一个Linear
- `main.py`: 包含训练、生成、提供命令行接口等，跑模型的代码都在这里；main.ipynb可以拆分自己调试看看细节
- `utils.py`: 可视化相关，拿来就用了，原理不是很清楚
- `tang.npz`: 诗歌的numpy压缩数据，包含57,580首诗，每首诗最大长度是125；词汇量大小8293


## 环境配置

- 安装[PyTorch](http://pytorch.org)
- 安装第三方依赖

```Python
pip install -r requirements.txt
```

- 启动visodm
```Bash
 python -m visdom.server
```
或者
```Bash
nohup python -m visdom.server &
```

visdom的可视化界面地址是：http://host:8097/

## 训练

训练的命令如下：

```Bash
python main.py train --plot-every=150\
					 --batch-size=128\
                     --pickle-path='tang.npz'\
                     --lr=1e-3 \
                     --env='poetry3' \
                     --epoch=50
```

命令行选项：
```Python
    data_path = 'data/' # 诗歌的文本文件存放路径
    pickle_path= 'tang.npz' # 预处理好的二进制文件 
    author = None # 只学习某位作者的诗歌
    constrain = None # 长度限制
    category = 'poet.tang' # 类别，唐诗还是宋诗歌(poet.song)
    lr = 1e-3 
    weight_decay = 1e-4
    use_gpu = True
    epoch = 20  
    batch_size = 128
    maxlen = 125 # 超过这个长度的之后字被丢弃，小于这个长度的在前面补空格
    plot_every = 20 # 每20个batch 可视化一次
    use_env = True # 是否使用visodm
    env='poetry' # visdom env
    max_gen_len = 200 # 生成诗歌最长长度
    debug_file='/tmp/debug'
    model_path=None # 预训练模型路径
    prefix_words = '细雨鱼儿出,微风燕子斜。' # 不是诗歌的组成部分，用来控制生成诗歌的意境
    start_words='闲云潭影日悠悠' # 诗歌开始
    acrostic = False # 是否是藏头诗
    model_prefix = 'checkpoints/tang' # 模型保存路径

```

## 生成诗歌

作者提供了预训练好的模型，`checkpoints/tang_199.pth`，用以生成诗歌

生成藏头诗的命令如下：

```Bash
python  main.py gen  --model-path='checkpoints/tang_199.pth' \
       --pickle-path='tang.npz' \
       --start-words='深度学习' \
       --prefix-words='江流天地外，山色有无中。' \
       --acrostic=True\
       ----use-gpu=True
深山万里外，不见三月风。度月出门夜，复来无定同。学道到山中，出门望层穹。习习一为内，有路不逡长。
```

生成其它诗歌的命令如下：

```Bash
python2 main.py gen  --model-path='model.pth' 
					 --pickle-path='tang.npz' 
					 --start-words='江流天地外，' # 诗歌的开头
					 --prefix-words='郡邑浮前浦，波澜动远空。' 
江流天地外，舟楫夜行东。相望登楼望，遥思北阙空。望云凝万象，碎雪冒秋风。地梗非归觐，戎书已再通。封疆曾屡辔，畎浍述有风。宁知农氏別，祗此浙江东。览镜知难遂，怀贤趣未终。远峰疑欲泛，瞻望忽无穷。飞鸟慙王粲，王公叙谢公。小臣叨侍从，题组赋成觞。具盗休民友，悠悠酷祀忠。
```

```Bash
python  main.py gen  --model-path='checkpoints/tang_199.pth' \
       --pickle-path='tang.npz' \
       --start-words='深度学习' \
       --prefix-words='江流天地外，山色有无中。' \
       --use-gpu=True
深度学习情不胜，人间多在无由见。青丝缕散千年春，绿褥三杯一两人。醉后不知春思后，一双双燕不同人。花落门前江上诉，月明台上看春色。嫁得蛾眉照素裙，妾身不见裁梳舞。画娥蝴蝶不相逢，惆怅人间不相见。愿君一夜相思心，一笑相思一千里。此时相思不相见，况复琵琶弦无说。
```