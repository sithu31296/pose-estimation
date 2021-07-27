import numpy as np
from openpifpaf.functional import scalar_square_add_gauss_with_max, scalar_values, Occupancy, grow_connection_blend



def CifHr(field, meta):
    neighbors = 16
    v_threshold = 0.1
    C, _, H, W = field.shape
    shape = (C, int((H-1) * meta.stride + 1), int((W-1) * meta.stride + 1))
    ta = np.zeros(shape, dtype=np.float32)

    for t, p in zip(ta, field):
        p = p[:, p[0] > v_threshold]
        v, x, y, _, scale = p
        x = x * meta.stride
        y = y * meta.stride
        sigma = np.maximum(1.0, 0.5 * scale * meta.stride)
        scalar_square_add_gauss_with_max(t, x, y, sigma, v / neighbors, truncate=1.0)

    return ta


def CifSeeds(cifhr, field, meta):
    threshold = 0.5
    seeds = []

    for field_i, p in enumerate(field):
        p = p[:, p[0] > threshold]
        c, x, y, _, s = p
        v = scalar_values(cifhr[field_i], x * meta.stride, y * meta.stride, default=0.0)
        v = 0.9 * v + 0.1 * c
        m = v > threshold
        x, y, v, s = x[m] * meta.stride, y[m] * meta.stride, v[m], s[m] * meta.stride

        for vv, xx, yy, ss in zip(v, x, y, s):
            seeds.append((vv, field_i, xx, yy, ss))

    return sorted(seeds, reverse=True)


class CafScored:
    score_th = 0.2
    cif_floor = 0.1
    def __init__(self, cifhr, field, meta) -> None:
        self.cifhr = cifhr
        self.forward = [np.empty((9, 0), dtype=field.dtype) for _ in field]
        self.backward = [np.empty((9, 0), dtype=field.dtype) for _ in field]

        for caf_i, nine in enumerate(field):
            assert nine.shape[0] == 9
            mask = nine[0] > self.score_th

            if not np.any(mask):
                continue

            nine = nine[:, mask]

            if meta.decoder_min_distance:
                dist = np.linalg.norm(nine[1:3] - nine[3:5], axis=0)
                mask_dist = dist > meta.decoder_min_distance / meta.stride
                nine = nine[:, mask_dist]

            if meta.decoder_max_distance:
                dist = np.linalg.norm(nine[1:3] - nine[3:5], axis=0)
                mask_dist = dist < meta.decoder_max_distance / meta.stride
                nine = nine[:, mask_dist]

            nine[(1, 2, 3, 4, 5, 6, 7, 8), :] *= meta.stride

            nine_b = np.copy(nine[(0, 3, 4, 1, 2, 6, 5, 8, 7), :])
            nine_f = np.copy(nine)

            nine_b = self.rescore(nine_b, meta.skeleton[caf_i][0] - 1)
            nine_f = self.rescore(nine_f, meta.skeleton[caf_i][1] - 1)

            self.backward[caf_i] = np.concatenate((self.backward[caf_i], nine_b), axis=1)
            self.forward[caf_i] = np.concatenate((self.forward[caf_i], nine_f), axis=1)

    def rescore(self, nine, joint_t):
        if self.cif_floor < 1.0 and joint_t < len(self.cifhr):
            cifhr_t = scalar_values(self.cifhr[joint_t], nine[3], nine[4], default=0.0)
            nine[0] *= self.cif_floor + (1.0 - self.cif_floor) * cifhr_t
        return nine[:, nine[0] > self.score_th]

    def directed(self, caf_i, forward):
        if forward:
            return self.forward[caf_i], self.backward[caf_i]
        return self.backward[caf_i], self.forward[caf_i]


def nms_keypoints(anns):
    instance_threshold = 0.15
    keypoint_threshold = 0.15
    
    for ann in anns:
        ann.data[ann.data[:, 2] < keypoint_threshold] = 0.0
    anns = [ann for ann in anns if ann.score >= instance_threshold]

    if not anns:
        return anns

    # +1 for rounding up
    max_y = int(max(np.max(ann.data[:, 1]) for ann in anns) + 1)
    max_x = int(max(np.max(ann.data[:, 0]) for ann in anns) + 1)
    # +1 because non-inclusive boundary
    shape = (len(anns[0].data), max(1, max_y + 1), max(1, max_x + 1))
    occupied = Occupancy(shape, 2, min_scale=4)

    anns = sorted(anns, key=lambda a: -a.score)
    for ann in anns:
        assert ann.joint_scales is not None
        assert len(occupied) == len(ann.data)
        for f, (xyv, joint_s) in enumerate(zip(ann.data, ann.joint_scales)):
            v = xyv[2]
            if v == 0.0:
                continue

            if occupied.get(f, xyv[0], xyv[1]):
                xyv[2] *= 0.0
            else:
                occupied.set(f, xyv[0], xyv[1], joint_s)  # joint_s = 2 * sigma

    for ann in anns:
        ann.data[ann.data[:, 2] < keypoint_threshold] = 0.0
    anns = [ann for ann in anns if ann.score >= instance_threshold]
    anns = sorted(anns, key=lambda a: -a.score)

    return anns



class Annotation:
    def __init__(self, keypoints, skeleton, score_weights=None):
        self.skeleton = skeleton
        self.score_weights = score_weights
        self.data = np.zeros((len(keypoints), 3), dtype=np.float32)
        self.joint_scales = np.zeros((len(keypoints),), dtype=np.float32)
        self.decoding_order = []
        self.frontier_order = []

        self.score_weights = np.asarray(self.score_weights)
        self.score_weights /= np.sum(self.score_weights)

    def add(self, joint_i, xyv):
        self.data[joint_i] = xyv
        return self

    @property
    def score(self):
        v = self.data[:, 2]
        return np.sum(self.score_weights * np.sort(v)[::-1])
