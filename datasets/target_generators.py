import numpy as np


class HeatmapGenerator:
    """Generate heatmaps for bottom-up models
    """
    def __init__(self, size: int, num_joints: int, sigma: int = -1) -> None:
        self.size = size
        self.num_joints = num_joints
        self.sigma = self.size / 64 if sigma < 0 else sigma

        x = np.arange(0, 6 * sigma + 3, 1, dtype=np.float32)
        y = x[:, None]
        x0 = y0 = 3 * self.sigma + 1
        self.g = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * self.sigma**2))

    def __call__(self, joints):
        heat_maps = np.zeros((self.num_joints, self.size, self.size), dtype=np.float32)

        for joint in joints:
            for idx, (x, y, v) in enumerate(joint):
                x, y = tuple(map(int, (x, y)))
                if v > 0 and 0 <= x < self.size and 0 <= y < self.size:
                    x1, y1 = tuple(map(lambda a: int(np.floor(a - 3 * self.sigma - 1)), (x, y)))
                    x2, y2 = tuple(map(lambda a: int(np.ceil(a + 3 * self.sigma + 2)), (x, y)))
                    
                    c, d = max(0, -x1), min(x2, self.size) - x1
                    a, b = max(0, -y1), min(y2, self.size) - y1

                    x1, x2 = max(0, x1), min(x2, self.size)
                    y1, y2 = max(0, y1), min(y2, self.size)

                    heat_maps[idx, y1:y2, x1:x2] = np.maximum(heat_maps[idx, y1:y2, x1:x2], self.g[a:b, c:d])

        return heat_maps


class JointsEncoder:
    """Encodes the visible joints into (coordinates, score)
    """
    def __init__(self, max_num_people, num_joints, size, tag_per_joint) -> None:
        self.max_num_people = max_num_people
        self.num_joints = num_joints
        self.size = size
        self.tag_per_joint = tag_per_joint

    def __call__(self, joints):
        visible_kpts = np.zeros((self.max_num_people, self.num_joints, 2), dtype=np.float32)

        for i, joint in enumerate(joints):
            total = 0
            for idx, (x, y, v) in enumerate(joint):
                x, y = tuple(map(int, (x, y)))
                if v > 0 and 0 <= x < self.size and 0 <= y < self.size:
                    if self.tag_per_joint:
                        visible_kpts[i][total] = (idx * self.size**2 + y * self.size + x), 1
                    else:
                        visible_kpts[i][total] = (y * self.size + x), 1
        return visible_kpts


class PAFGenerator:
    """Generate part affinity fields
    """
    def __init__(self, size, limb_width, skeleton) -> None:
        self.size = size
        self.limb_width = limb_width
        self.skeleton = skeleton

    def accumulate_paf_map(self, pafs, src, dst, count):
        """Accumulate PAF between two given joints
        """
        limb_vec = dst - src
        norm = np.linalg.norm(limb_vec)

        if norm == 0:
            unit_limb_vec = np.zeros(2)
        else:
            unit_limb_vec = limb_vec / norm

        x1, y1 = src
        x2, y2 = dst

        min_x, min_y = tuple(map(lambda a, b: int(max(np.floor(min(a, b) - self.limb_width), 0)), [(x1, x2), (y1, y2)]))
        max_x = int(min(np.ceil(max(x1, x2) + self.limb_width), self.size - 1))
        max_y = int(min(np.ceil(max(y1, y2) + self.limb_width), self.size + 1))

        range_x = list(range(min_x, max_x + 1, 1))
        range_y = list(range(min_y, max_y + 1, 1))
        xx, yy = np.meshgrid(range_x, range_y)
        delta_x = xx - x1
        delta_y = yy - y1
        dist = np.abs(delta_x * unit_limb_vec[1] - delta_y * unit_limb_vec[0])
        mask = np.zeros_like(count, dtype=bool)
        mask[xx, yy] = dist < self.limb_width

        pafs[0, mask] += unit_limb_vec[0]
        pafs[1, mask] += unit_limb_vec[1]
        count += mask

    def __call__(self, joints):
        pafs = np.zeros((len(self.skeleton)*2, self.size, self.size), dtype=np.float32)

        for idx, (j1, j2) in enumerate(self.skeleton):
            count = np.zeros((self.size, self.size), dtype=np.float32)
            for joint in joints:
                x1, y1, v1 = joint[j1 - 1]
                x2, y2, v2 = joint[j2 - 1]
                if v1 > 0 and v2 > 0:
                    self.accumulate_paf_map(pafs[2*idx:2*idx+2], (x1, y1), (x2, y2), count)

            pafs[2*idx:2*idx+2] /= np.maximum(count, 1)

        return pafs

