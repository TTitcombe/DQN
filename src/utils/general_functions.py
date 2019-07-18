"""Functions which don't belong to any class/particular usage"""

import numpy as np
import os
import torch


def create_dir(save_path):
    path = ""
    for directory in os.path.split(save_path):
        path = os.path.join(path, directory)
        if not os.path.exists(path):
            os.mkdir(path)


def torch_from_frame(frame, device):
    frame = torch.from_numpy(np.ascontiguousarray(frame, dtype=np.float32))
    return frame.unsqueeze(0).to(device)
