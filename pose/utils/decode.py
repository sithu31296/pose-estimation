import math
import torch
import numpy as np
from torch import Tensor
from .utils import get_affine_transform


def get_simdr_final_preds(pred_x: Tensor, pred_y: Tensor, center, scale, image_size):
    B, C, _ = pred_x.shape
    pred_x, pred_y = pred_x.softmax(dim=2), pred_y.softmax(dim=2)
    pred_x, pred_y = pred_x.max(dim=2)[-1], pred_y.max(dim=2)[-1]

    coords = torch.ones(B, C, 2)
    coords[:, :, 0] = pred_x / 2
    coords[:, :, 1] = pred_y / 2

    coords = coords.cpu().numpy()
    preds = coords.copy()

    for i in range(B):
        preds[i] = transform_preds(coords[i], center[i], scale[i], image_size)
    return preds.astype(int)


def get_final_preds(heatmaps, center, scale):
    B, C, H, W = heatmaps.shape
    coords = get_max_preds(heatmaps)

    for n in range(B):
        for p in range(C):
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

    for i in range(B):
        preds[i] = transform_preds(coords[i], center[i], scale[i], [W, H])

    return preds.astype(int)


def get_max_preds(heatmaps: np.ndarray):
    B, C, _, W = heatmaps.shape
    heatmaps = heatmaps.reshape((B, C, -1))
    idx = np.argmax(heatmaps, axis=2).reshape((B, C, 1))
    maxvals = np.amax(heatmaps, axis=2).reshape((B, C, 1))
    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)
    preds[:, :, 0] = preds[:, :, 0] % W
    preds[:, :, 1] = preds[:, :, 1] // W
    preds = np.where(np.tile(maxvals, (1, 1, 2)) > 0.0, preds, -1)
    return preds


def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, output_size, inv=True)
    for p in range(coords.shape[0]):
        target_coords[p, :2] = affine_transform(coords[p, :2], trans)
    return target_coords


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt