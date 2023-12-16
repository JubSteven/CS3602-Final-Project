import os, shutil
from tensorboardX import SummaryWriter
root_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time

date = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())


def make_path(args):
    return (
        f"{args.expri}_{date}_max_epoch={args.max_epoch}_bs={args.batch_size}_lr={args.lr}_decay={args.decay_step}_gamma={args.gamma}"
        f"_encoder={args.encoder_cell}_dropout={args.dropout}"
    )


def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path, 0o777,exist_ok=True)  # 0o777 means the mode of the folder, which is the permission of the folder, could be read, write and execute.


def visualizer(args, clear_visualizer=True):
    save_path = make_path(args)
    # filewriter_path = config['visual_base']+args.savepath+'/'

    filewriter_path = os.path.join(root_path, args.visualize_path, save_path)
    if clear_visualizer and os.path.exists(filewriter_path):  # 删掉以前的summary，以免重合
        shutil.rmtree(filewriter_path)
    makedir(filewriter_path)
    writer = SummaryWriter(filewriter_path, comment='visualizer')
    return writer
