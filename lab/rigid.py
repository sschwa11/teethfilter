import glob
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")

SEED_RIGID = [10, 151, 9, 8, 168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 133, 362, 33, 263, 127, 356, 234, 454, 21, 251, 109, 338,
              67, 297, 103, 332, 54, 284, 162, 389, 71, 301, 68, 298, 104, 333, 69, 299, 108, 337, 107, 336, 55, 285, 193, 417,
              122, 351, 245, 465, 188, 412, 114, 343, 217, 437, 198, 420, 131, 360, 49, 279, 48, 278, 219, 439, 115, 344, 45, 275,
              44, 274, 220, 440, 134, 363, 236, 456, 3, 248, 51, 281, 196, 419, 174, 399]


def umeyama(src, dst, w=None):
    if w is None:
        w = np.ones(len(src))
    w = w / w.sum()
    ms = (w[:, None] * src).sum(0)
    md = (w[:, None] * dst).sum(0)
    a = src - ms
    b = dst - md
    cov = (w[:, None] * b).T @ a
    u, d, vt = np.linalg.svd(cov)
    s_ = np.ones(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s_[-1] = -1
    r = u @ np.diag(s_) @ vt
    var = (w * (a ** 2).sum(1)).sum()
    s = (d * s_).sum() / var
    t = md - s * r @ ms
    return s, r, t


def load_all():
    out = {}
    for p in sorted(glob.glob(os.path.join(CACHE, "*.npz"))):
        d = np.load(p, allow_pickle=True)
        if d["ok"].sum() < 60:
            continue
        out[os.path.splitext(os.path.basename(p))[0]] = {k: d[k] for k in d.files}
    return out


def gpa(shapes, idx, iters=6):
    ref = shapes[0][idx].copy()
    ref -= ref.mean(0)
    for _ in range(iters):
        al = []
        for sh in shapes:
            s, r, t = umeyama(sh[idx], ref)
            al.append((s * (r @ sh.T)).T + t)
        al = np.array(al)
        ref_full = al.mean(0)
        ref = ref_full[idx]
    return ref_full, al


def build_reference(data, max_frames=4000, seed=0):
    rng = np.random.default_rng(seed)
    shapes = []
    for k, d in data.items():
        v = np.flatnonzero(d["ok"])
        shapes.append(d["pts"][v].astype(np.float64))
    shapes = np.concatenate(shapes)
    sel = rng.choice(len(shapes), min(max_frames, len(shapes)), replace=False)
    shapes = shapes[sel]
    iod = np.linalg.norm(shapes[:, 33] - shapes[:, 263], axis=1)
    shapes = shapes / iod[:, None, None] * 100.0
    idx = np.array(SEED_RIGID)
    ref, al = gpa(list(shapes), idx)
    resid = np.linalg.norm(al - ref[None], axis=2)
    spread = np.median(resid, axis=0)
    return ref, spread, al


CW, CH = 320, 260


class DentalFrame:
    def __init__(self, ref, idx):
        ex = ref[263] - ref[33]
        ex /= np.linalg.norm(ex)
        ey = ref[152] - ref[10]
        ey -= ex * np.dot(ey, ex)
        ey /= np.linalg.norm(ey)
        ez = np.cross(ex, ey)
        self.basis = np.stack([ex, ey, ez])
        self.ref = ref
        self.idx = np.asarray(idx)
        c = (ref[13] + ref[14]) / 2
        self.center = c + 0.08 * np.linalg.norm(ref[263] - ref[33]) * ez
        self.k = np.linalg.norm(ref[308] - ref[78]) / 200.0

    def pose(self, pts):
        return umeyama(self.ref[self.idx], pts[self.idx].astype(np.float64))

    def affine(self, pose):
        s, r, t = pose
        ex, ey, _ = self.basis
        cols = np.stack([ex * self.k, ey * self.k], 1)
        o = self.center - (CW / 2) * self.k * ex - (CH / 2) * self.k * ey
        a = np.zeros((2, 3))
        a[:, :2] = (s * r @ cols)[:2]
        a[:, 2] = (s * r @ o + t)[:2]
        return a

    def to_crop(self, pose, pts):
        a = self.affine(pose)
        m = np.vstack([a, [0, 0, 1]])
        inv = np.linalg.inv(m)
        return pts[:, :2] @ inv[:2, :2].T + inv[:2, 2]

    def warp(self, frame, pose, size=(CW, CH)):
        return cv2.warpAffine(frame, self.affine(pose).astype(np.float32), size, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
