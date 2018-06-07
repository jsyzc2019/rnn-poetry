class Config(object):
    data_path = 'data/'
    pickle_path = 'tang.npz'
    author = None
    constrain = None
    category = 'poet.tang'  # 类别：poet.tang poet.song 唐诗宋词
    lr = 1e-3
    use_gpu = True
    epoch = 200
    batch_size = 128
    maxlen = 125
    plot_every = 20
    embedding_dim = 128
    hidden_dim = 256
    use_env = True
    env = 'poetry'
    max_gen_len = 200
    debug_file = '/tmp/debug'
    model_path = None
    
    prefix_words = '细雨鱼儿出。微风燕子斜。'  # 用来控制诗歌的意境的
    start_words = '闲云潭影日悠悠'  # 诗歌开始的部分
    acrostic = False  # 是否是藏头诗，True 是，False 否
    model_prefix = 'checkpoints/tang'  # 保存模型路径前缀