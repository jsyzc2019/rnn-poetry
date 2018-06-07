import os
import numpy as np

def get_data(opt):
    if os.path.exists(opt.pickle_path):
        datas = np.load(opt.pickle_path)
        data, word2ix, ix2word = datas['data'], datas['word2ix'].item(), datas['ix2word'].item()
        return data, word2ix, ix2word