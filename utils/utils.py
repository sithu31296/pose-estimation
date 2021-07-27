import cv2
import numpy as np
from matplotlib import cm, collections
import matplotlib.pyplot as plt
import random
import torch
import numpy as np
import random
import time
import os
from pathlib import Path
from torch.backends import cudnn
from torch import nn
from torch.autograd import profiler
from typing import Union
from torch import distributed as dist


def fix_seeds(seed: int = 123) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def setup_cudnn() -> None:
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    cudnn.benchmark = True
    cudnn.deterministic = False

def time_synchronized() -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

def get_model_size(model: Union[nn.Module, torch.jit.ScriptModule]):
    tmp_model_path = Path('temp.p')
    if isinstance(model, torch.jit.ScriptModule):
        torch.jit.save(model, tmp_model_path)
    else:
        torch.save(model.state_dict(), tmp_model_path)
    size = tmp_model_path.stat().st_size
    os.remove(tmp_model_path)
    return size / 1e6   # in MB

@torch.no_grad()
def test_model_latency(model: nn.Module, inputs: torch.Tensor, use_cuda: bool = False) -> float:
    with profiler.profile(use_cuda=use_cuda) as prof:
        _ = model(inputs)
    return prof.self_cpu_time_total / 1000  # ms

def setup_ddp() -> None:
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ(['LOCAL_RANK']))

    torch.cuda.set_device(gpu)
    dist.init_process_group('nccl', world_size=world_size, rank=rank)
    dist.barrier()

    return rank, world_size, gpu

def cleanup_ddp():
    dist.destroy_process_group()

def draw_keypoints(img, keypoints, skeletons):
    if keypoints == []: return img
    image = img.copy()
    for kpts in keypoints:
        for x, y, v in kpts:
            if v == 2:
                cv2.circle(image, (x, y), 4, (255, 0, 0), 2)
        for kid1, kid2 in skeletons:
            x1, y1, v1 = kpts[kid1-1]
            x2, y2, v2 = kpts[kid2-1]
            if v1 == 2 and v2 == 2:
                cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)   
    return image 
