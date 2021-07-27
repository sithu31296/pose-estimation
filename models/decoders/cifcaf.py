import numpy as np
import heapq
from collections import defaultdict
from .decode_utils import CifHr, CifSeeds, CafScored, nms_keypoints, Annotation, Occupancy, grow_connection_blend
    

class CifCaf:
    """Generate CifCaf poses from fields
    """
    connection_method = 'blend'
    keypoint_threshold = 0.15
    keypoint_threshold_rel = 0.5

    def __init__(self, cif_meta, caf_meta) -> None:
        self.cif_meta = cif_meta
        self.caf_meta = caf_meta
        self.skeleton_m1 = np.asarray(caf_meta.skeleton) - 1
        self.keypoints = cif_meta.keypoints
        self.out_skeleton = caf_meta.skeleton
        self.score_weights = cif_meta.score_weights
        self.confidence_scales = caf_meta.decoder_confidence_scales

        # prefer decoders with more keypoints and associations
        self.priority = cif_meta.n_fields / 1000.0
        self.priority += caf_meta.n_fields / 1000.0

        self.by_source = defaultdict(dict)
        for caf_i, (j1, j2) in enumerate(self.skeleton_m1):
            self.by_source[j1][j2] = (caf_i, True)
            self.by_source[j2][j1] = (caf_i, False)


    def __call__(self, fields):
        cifhr = CifHr(fields[0], self.cif_meta)
        seeds = CifSeeds(cifhr, fields[0], self.cif_meta)
        caf_scored = CafScored(cifhr, fields[1], self.caf_meta)
        occupied = Occupancy(cifhr.shape, 2, min_scale=4)

        annotations = []

        def mark_occupied(ann):
            joint_is = np.flatnonzero(ann.data[:, 2])
            for joint_i in joint_is:
                width = ann.joint_scales[joint_i]
                occupied.set(joint_i, ann.data[joint_i, 0], ann.data[joint_i, 1], width)

        for v, f, x, y, s in seeds:
            if occupied.get(f, x, y):
                continue

            ann = Annotation(self.keypoints, self.out_skeleton, self.score_weights).add(f, (x, y, v))
            ann.joint_scales[f] = s
            self._grow(ann, caf_scored)
            annotations.append(ann)
            mark_occupied(ann)

        annotations = nms_keypoints(annotations)

        return annotations

    def connection_value(self, ann, caf_scored, start_i, end_i):
        caf_i, forward = self.by_source[start_i][end_i]
        caf_f, caf_b = caf_scored.directed(caf_i, forward)
        xyv = ann.data[start_i]
        xy_scale_s = max(0.0, ann.joint_scales[start_i])

        new_xysv = grow_connection_blend(caf_f, xyv[0], xyv[1], xy_scale_s, False)
        if new_xysv[3] == 0.0:
            return 0.0, 0.0, 0.0, 0.0
        keypoint_score = np.sqrt(new_xysv[3] * xyv[2])  # geometric mean
        if keypoint_score < self.keypoint_threshold:
            return 0.0, 0.0, 0.0, 0.0
        if keypoint_score < xyv[2] * self.keypoint_threshold_rel:
            return 0.0, 0.0, 0.0, 0.0
        xy_scale_t = max(0.0, new_xysv[2])

        # reverse match
        reverse_xyv = grow_connection_blend(caf_b, new_xysv[0], new_xysv[1], xy_scale_t, False)
        if reverse_xyv[2] == 0.0:
            return 0.0, 0.0, 0.0, 0.0
        if abs(xyv[0] - reverse_xyv[0]) + abs(xyv[1] - reverse_xyv[1]) > xy_scale_s:
            return 0.0, 0.0, 0.0, 0.0

        return (new_xysv[0], new_xysv[1], new_xysv[2], keypoint_score)
    
    def _grow(self, ann, caf_scored):
        frontier = []
        in_frontier = set()

        def add_to_frontier(start_i):
            for end_i, (caf_i, _) in self.by_source[start_i].items():
                if ann.data[end_i, 2] > 0.0:
                    continue
                if (start_i, end_i) in in_frontier:
                    continue

                max_possible_score = np.sqrt(ann.data[start_i, 2])
                if self.confidence_scales is not None:
                    max_possible_score *= self.confidence_scales[caf_i]
                heapq.heappush(frontier, (-max_possible_score, None, start_i, end_i))
                in_frontier.add((start_i, end_i))
                ann.frontier_order.append((start_i, end_i))

        def frontier_get():
            while frontier:
                entry = heapq.heappop(frontier)
                if entry[1] is not None:
                    return entry

                _, __, start_i, end_i = entry
                if ann.data[end_i, 2] > 0.0:
                    continue

                new_xysv = self.connection_value(ann, caf_scored, start_i, end_i)
                if new_xysv[3] == 0.0:
                    continue
                score = new_xysv[3]
                if self.confidence_scales is not None:
                    caf_i, _ = self.by_source[start_i][end_i]
                    score = score * self.confidence_scales[caf_i]
                heapq.heappush(frontier, (-score, new_xysv, start_i, end_i))

        # seeding the frontier
        for joint_i in np.flatnonzero(ann.data[:, 2]):
            add_to_frontier(joint_i)

        while True:
            entry = frontier_get()
            if entry is None:
                break

            _, new_xysv, jsi, jti = entry
            if ann.data[jti, 2] > 0.0:
                continue

            ann.data[jti, :2] = new_xysv[:2]
            ann.data[jti, 2] = new_xysv[3]
            ann.joint_scales[jti] = new_xysv[2]
            ann.decoding_order.append((jsi, jti, np.copy(ann.data[jsi]), np.copy(ann.data[jti])))
            add_to_frontier(jti)

        