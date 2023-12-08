import os, shutil
from tensorboardX import SummaryWriter

import numpy as np
import time

date = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

from args import opts

def make_path():
    return (
        f"{date}_max_epoch={opts.max_epochs}_bs={opts.batch_size}_lr={opts.lr}_seed={opts.seed}_device={opts.device}"
        f"_encoder={opts.encoder_cell}_dropout={opts.dropout}_embed={opts.embed_size}_hidden={opts.hidden_size}_layer={opts.num_layer}"
    )


def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path, 0o777)  # 0o777 means the mode of the folder, which is the permission of the folder, could be read, write and execute.


def visualizer(dir="tensorBoard", clear_visualizer=True):
    save_path = make_path()
    # filewriter_path = config['visual_base']+opts.savepath+'/'

    filewriter_path = os.path.join(dir, save_path)
    if opts.clear_visualizer and os.path.exists(filewriter_path):  # 删掉以前的summary，以免重合
        shutil.rmtree(filewriter_path)
    makedir(filewriter_path)
    writer = SummaryWriter(filewriter_path, comment='visualizer')
