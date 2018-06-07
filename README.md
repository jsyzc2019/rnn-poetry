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
