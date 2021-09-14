import cv2
import numpy as np
import random
import torch
import time
import os
import math
from torch import Tensor
from torchvision import io, ops
from threading import Thread
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

def draw_coco_keypoints(img, keypoints, skeletons):
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


def draw_keypoints(img, keypoints, skeletons):
    if keypoints == []: return img
    for kpts in keypoints:
        for x, y in kpts:
            cv2.circle(img, (x, y), 4, (255, 0, 0), 2, cv2.LINE_AA)
        for kid1, kid2 in skeletons:
            cv2.line(img, kpts[kid1-1], kpts[kid2-1], (0, 255, 0), 2, cv2.LINE_AA)   


class WebcamStream:
    def __init__(self, src=0) -> None:
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        assert self.cap.isOpened(), f"Failed to open webcam {src}"
        _, self.frame = self.cap.read()
        Thread(target=self.update, args=([]), daemon=True).start()

    def update(self):
        while self.cap.isOpened():
            _, self.frame = self.cap.read()

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1

        if cv2.waitKey(1) == ord('q'):
            self.stop()

        return self.frame.copy()

    def stop(self):
        self.cap.release()
        cv2.destroyAllWindows()
        raise StopIteration

    def __len__(self):
        return 0


class VideoReader:
    def __init__(self, video: str):
        self.frames, _, info = io.read_video(video, pts_unit='sec')
        self.fps = info['video_fps']

        print(f"Processing '{video}'...")
        print(f"Total Frames: {len(self.frames)}")
        print(f"Video Size  : {list(self.frames.shape[1:-1])}")
        print(f"Video FPS   : {self.fps}")

    def __iter__(self):
        self.count = 0
        return self

    def __len__(self):
        return len(self.frames)

    def __next__(self):
        if self.count == len(self.frames):
            raise StopIteration
        frame = self.frames[self.count]
        self.count += 1
        return frame


class VideoWriter:
    def __init__(self, file_name, fps):
        self.fname = file_name
        self.fps = fps
        self.frames = []

    def update(self, frame):
        if isinstance(frame, np.ndarray):
            frame = torch.from_numpy(frame)
        self.frames.append(frame)

    def write(self):
        print(f"Saving video to '{self.fname}'...")
        io.write_video(self.fname, torch.stack(self.frames), self.fps)


class FPS:
    def __init__(self, avg=10) -> None:
        self.accum_time = 0
        self.counts = 0
        self.avg = avg

    def synchronize(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def start(self):
        self.synchronize()
        self.prev_time = time.time()

    def stop(self, debug=True):
        self.synchronize()
        self.accum_time += time.time() - self.prev_time
        self.counts += 1
        if self.counts == self.avg:
            self.fps = round(self.counts / self.accum_time)
            if debug: print(f"FPS: {self.fps}")
            self.counts = 0
            self.accum_time = 0

def letterbox(img, new_shape=(640, 640)):
    H, W = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / H, new_shape[1] / W)
    nH, nW = round(H * r), round(W * r)
    pH, pW = np.mod(new_shape[0] - nH, 32) / 2, np.mod(new_shape[1] - nW, 32) / 2

    if (H, W) != (nH, nW):
        img = cv2.resize(img, (nW, nH), interpolation=cv2.INTER_LINEAR)

    top, bottom = round(pH - 0.1), round(pH + 0.1)
    left, right = round(pW - 0.1), round(pW + 0.1)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img


def scale_boxes(boxes, orig_shape, new_shape):
    H, W = orig_shape
    nH, nW = new_shape
    gain = min(nH / H, nW / W)
    pad = (nH - H * gain) / 2, (nW - W * gain) / 2

    boxes[:, ::2] -= pad[1]
    boxes[:, 1::2] -= pad[0]
    boxes[:, :4] /= gain
    
    boxes[:, ::2].clamp_(0, orig_shape[1])
    boxes[:, 1::2].clamp_(0, orig_shape[0])
    return boxes.round()


def xywh2xyxy(x):
    boxes = x.clone()
    boxes[:, 0] = x[:, 0] - x[:, 2] / 2
    boxes[:, 1] = x[:, 1] - x[:, 3] / 2
    boxes[:, 2] = x[:, 0] + x[:, 2] / 2
    boxes[:, 3] = x[:, 1] + x[:, 3] / 2
    return boxes


def xyxy2xywh(x):
    y = x.clone()
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, classes=None):
    candidates = pred[..., 4] > conf_thres 

    max_wh = 4096
    max_nms = 30000
    max_det = 300

    output = [torch.zeros((0, 6), device=pred.device)] * pred.shape[0]

    for xi, x in enumerate(pred):
        x = x[candidates[xi]]

        if not x.shape[0]: continue

        # compute conf
        x[:, 5:] *= x[:, 4:5]   # conf = obj_conf * cls_conf

        # box
        box = xywh2xyxy(x[:, :4])

        # detection matrix nx6
        conf, j = x[:, 5:].max(1, keepdim=True)
        x = torch.cat([box, conf, j.float()], dim=1)[conf.view(-1) > conf_thres]

        # filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # check shape
        n = x.shape[0]
        if not n: 
            continue
        elif n > max_nms:
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        # batched nms
        c = x[:, 5:6] * max_wh
        boxes, scores = x[:, :4] + c, x[:, 4]
        keep = ops.nms(boxes, scores, iou_thres)

        if keep.shape[0] > max_det:
            keep = keep[:max_det]

        output[xi] = x[keep]

    return output


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt


def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, :2] = affine_transform(coords[p, :2], trans)
    return target_coords


def get_max_preds(heatmaps: np.ndarray):
    """Get predictions from score maps
    heatmaps: (ndarray) in shape[B, C, H, W]
    """
    B, C, _, W = heatmaps.shape
    heatmaps = heatmaps.reshape((B, C, -1))
    idx = np.argmax(heatmaps, axis=2).reshape((B, C, 1))
    maxvals = np.amax(heatmaps, axis=2).reshape((B, C, 1))
    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)
    preds[:, :, 0] = preds[:, :, 0] % W
    preds[:, :, 1] = preds[:, :, 1] // W
    preds = np.where(np.tile(maxvals, (1, 1, 2)) > 0.0, preds, -1)
    return preds


def get_final_preds(heatmaps, center, scale):
    B, C, H, W = heatmaps.shape
    coords = get_max_preds(heatmaps)

    for n in range(coords.shape[0]):
        for p in range(coords.shape[1]):
            hm = heatmaps[n][p]
            px = int(math.floor(coords[n][p][0] + 0.5))
            py = int(math.floor(coords[n][p][1] + 0.5))

            if 1 < px < W - 1 and 1 < py < H - 1:
                diff = np.array([
                    hm[py][px+1] - hm[py][px-1],
                    hm[py+1][px] - hm[py-1][px]
                ])
                coords[n][p] += np.sign(diff) * .25
    preds = coords.copy()

    for i in range(coords.shape[0]):
        preds[i] = transform_preds(coords[i], center[i], scale[i], [W, H])

    return preds.astype(int)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs
    return src_result

def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_affine_transform(center, scale, patch_size, rot=0, inv=0):
    shift = np.array([0, 0], dtype=np.float32)
    scale_tmp = scale * 200
    src_w = scale_tmp[0]
    dst_w = patch_size[0]
    dst_h = patch_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], dtype=np.float32)
    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)

    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def get_simdr_final_preds(pred_x: Tensor, pred_y: Tensor, center, scale, image_size):
    B, C, _ = pred_x.shape
    pred_x, pred_y = pred_x.softmax(dim=2), pred_y.softmax(dim=2)
    pred_x, pred_y = pred_x.max(dim=2, keepdim=True)[-1], pred_y.max(dim=2, keepdim=True)[-1]

    coords = torch.ones(B, C, 2)
    coords[:, :, 0] = torch.true_divide(pred_x, 2.0).squeeze()
    coords[:, :, 1] = torch.true_divide(pred_y, 2.0).squeeze()

    coords = coords.cpu().numpy()
    preds = coords.copy()

    for i in range(B):
        preds[i] = transform_preds(coords[i], center[i], scale[i], image_size)
    
    return preds.astype(int)