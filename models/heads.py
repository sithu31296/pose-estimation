import torch
import functools
import numpy as np
from torch import nn, Tensor
from torch.nn import functional as F
from typing import List, Tuple, ClassVar, List, Any
from dataclasses import dataclass, field

@functools.lru_cache(maxsize=16)
def index_field_torch(shape, *, device=None, unsqueeze=(0, 0)):
    assert len(shape) == 2
    xy = torch.empty((2, shape[0], shape[1]), device=device)
    xy[0] = torch.arange(shape[1], device=device)
    xy[1] = torch.arange(shape[0], device=device).unsqueeze(1)

    for dim in unsqueeze:
        xy = torch.unsqueeze(xy, dim)

    return xy


@dataclass
class Base:
    name: str
    dataset: str = 'cocokp'
    keypoints = [
        'nose',            # 1
        'left_eye',        # 2
        'right_eye',       # 3
        'left_ear',        # 4
        'right_ear',       # 5
        'left_shoulder',   # 6
        'right_shoulder',  # 7
        'left_elbow',      # 8
        'right_elbow',     # 9
        'left_wrist',      # 10
        'right_wrist',     # 11
        'left_hip',        # 12
        'right_hip',       # 13
        'left_knee',       # 14
        'right_knee',      # 15
        'left_ankle',      # 16
        'right_ankle',     # 17
    ]

    sigmas = [
        0.026,  # nose
        0.025,  # eyes
        0.025,  # eyes
        0.035,  # ears
        0.035,  # ears
        0.079,  # shoulders
        0.079,  # shoulders
        0.072,  # elbows
        0.072,  # elbows
        0.062,  # wrists
        0.062,  # wrists
        0.107,  # hips
        0.107,  # hips
        0.087,  # knees
        0.087,  # knees
        0.089,  # ankles
        0.089,  # ankles
    ]

    pose = np.array([
        [0.0, 9.3, 2.0],  # 'nose',            # 1
        [-0.35, 9.7, 2.0],  # 'left_eye',        # 2
        [0.35, 9.7, 2.0],  # 'right_eye',       # 3
        [-0.7, 9.5, 2.0],  # 'left_ear',        # 4
        [0.7, 9.5, 2.0],  # 'right_ear',       # 5
        [-1.4, 8.0, 2.0],  # 'left_shoulder',   # 6
        [1.4, 8.0, 2.0],  # 'right_shoulder',  # 7
        [-1.75, 6.0, 2.0],  # 'left_elbow',      # 8
        [1.75, 6.2, 2.0],  # 'right_elbow',     # 9
        [-1.75, 4.0, 2.0],  # 'left_wrist',      # 10
        [1.75, 4.2, 2.0],  # 'right_wrist',     # 11
        [-1.26, 4.0, 2.0],  # 'left_hip',        # 12
        [1.26, 4.0, 2.0],  # 'right_hip',       # 13
        [-1.4, 2.0, 2.0],  # 'left_knee',       # 14
        [1.4, 2.1, 2.0],  # 'right_knee',      # 15
        [-1.4, 0.0, 2.0],  # 'left_ankle',      # 16
        [1.4, 0.1, 2.0],  # 'right_ankle',     # 17
    ])

    n_confidences = 1
    training_weights: List[float] = None
    base_stride = 16
    stride = 8


@dataclass
class Cif(Base):
    "Head meta data for a Composite Intensity Field (CIF)"
    name: str = 'cif'
    draw_skeleton = [
        (16, 14), (14, 12), (17, 15), (15, 13), (12, 13), (6, 12), (7, 13),
        (6, 7), (6, 8), (7, 9), (8, 10), (9, 11), (2, 3), (1, 2), (1, 3),
        (2, 4), (3, 5), (4, 6), (5, 7),
    ]
    score_weights = [3.0] * 3 + [1.0] * (17 - 3)

    n_vectors = 1
    n_scales = 1
    n_fields = 17
    # vector_offsets = [True]
    decoder_min_scale = 0.0

    decoder_seed_mask: List[int] = None
    head_index = 0


@dataclass
class Caf(Base):
    "Head meta data for a Composite Association Field (CAF)"
    name: str = 'caf'
    skeleton = [
        (16, 14), (14, 12), (17, 15), (15, 13), (12, 13), (6, 12), (7, 13),
        (6, 7), (6, 8), (7, 9), (8, 10), (9, 11), (2, 3), (1, 2), (1, 3),
        (2, 4), (3, 5), (4, 6), (5, 7),
    ]
    sparse_skeleton: List[Tuple[int, int]] = None
    dense_to_sparse_radius: float = 2.0
    only_in_field_of_view: bool = False

    n_vectors = 2
    n_scales = 2
    n_fields = 19
    # vector_offsets = [True, True]
    decoder_min_distance = 0.0
    decoder_max_distance = float('inf')

    decoder_confidence_scales: List[float] = None
    head_index = 1
    


class CompositeField3(nn.Module):
    def __init__(self, meta: Base, in_features) -> None:
        super().__init__()
        self.meta = meta
        self.dropout = nn.Dropout2d(0.0)
        out_features = meta.n_fields * (meta.n_confidences + meta.n_vectors * 3 + meta.n_scales)
        self.conv = nn.Conv2d(in_features, out_features * 4, 1)
        self.upsample_op = nn.PixelShuffle(2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.dropout(x)
        x = self.conv(x)
        x = self.upsample_op(x)
        x = x[:, :, :x.shape[2]-1, :x.shape[3]-1]

        B, _, H, W = x.size()
        x = x.view(B, self.meta.n_fields, -1, H, W)

        if not self.training:
            # classification
            torch.sigmoid_(x[:, :, 0:self.meta.n_confidences])

            # regressions x: add index
            index_field = index_field_torch((H, W), device=x.device)
            for i in range(self.meta.n_vectors):
                x[:, :, self.meta.n_confidences + i * 2:self.meta.n_confidences + (i + 1) * 2] += index_field

            # scale
            first_scale_feature = self.meta.n_confidences + self.meta.n_vectors * 3
            x[:, :, first_scale_feature:first_scale_feature + self.meta.n_scales] = F.softplus(x[:, :, first_scale_feature:first_scale_feature + self.meta.n_scales])
        return x