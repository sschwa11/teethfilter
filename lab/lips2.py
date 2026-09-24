import cv2
import numpy as np

UPPER_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
LOWER_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
UPPER_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
LOWER_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]


def curve(cp, ids, width):
    p = cp[ids]
    o = np.argsort(p[:, 0])
    return np.interp(np.arange(width), p[o, 0], p[o, 1]).astype(np.float32), float(p[:, 0].min()), float(p[:, 0].max())


class Gauss:
    def __init__(self, x):
        self.mu = x.mean(0)
        c = np.cov(x.T) + np.eye(x.shape[1]) * 4.0
        self.ic = np.linalg.inv(c)
        self.ld = np.log(np.linalg.det(c))

    def logp(self, x):
        d = x - self.mu
        return -0.5 * (np.einsum("...i,ij,...j->...", d, self.ic, d) + self.ld)


def lip_prob(lab, cp, yu, yl, xl, xr):
    h, w = lab.shape[:2]
    xs = np.arange(w)
    cx = (xs > xl + 0.15 * (xr - xl)) & (xs < xr - 0.15 * (xr - xl))
    yuo, _, _ = curve(cp, UPPER_OUTER, w)
    ylo, _, _ = curve(cp, LOWER_OUTER, w)
    gy = np.arange(h)[:, None]
    up_band = (gy > yuo[None, :] + 0.35 * (yu - yuo)[None, :]) & (gy < yu[None, :] - 3) & cx[None, :]
    lo_band = (gy > yl[None, :] + 4) & (gy < yl[None, :] + 0.6 * (ylo - yl)[None, :]) & cx[None, :]
    inner = (gy > yu[None, :] + 3) & (gy < yl[None, :] - 3) & cx[None, :]
    lips = np.concatenate([lab[up_band], lab[lo_band]])
    ins = lab[inner]
    if len(lips) < 40:
        return None
    gl = Gauss(lips)
    lp = gl.logp(lab)
    if len(ins) < 40:
        thr = np.percentile(gl.logp(lips), 5)
        return 1 / (1 + np.exp(-(lp - thr)))
    li = gl.logp(ins)
    keep = li < np.percentile(gl.logp(lips), 10)
    ins = ins[keep] if keep.sum() > 40 else ins
    k = 3
    lab2 = ins.astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, lbl, _ = cv2.kmeans(lab2, k, None, crit, 2, cv2.KMEANS_PP_CENTERS)
    comps = [Gauss(ins[lbl.ravel() == j]) for j in range(k) if (lbl.ravel() == j).sum() > 15]
    lo = np.max(np.stack([c.logp(lab) for c in comps]), 0) if comps else lp - 10
    z = np.clip(lp - lo, -30, 30)
    return 1 / (1 + np.exp(-z))


def snap(score, y0, lo, hi, band, smooth):
    h, w = score.shape
    offs = np.arange(-band, band + 1)
    cols = np.arange(lo, hi)
    ys = np.clip(np.round(y0[cols][None, :] + offs[:, None]).astype(int), 0, h - 1)
    cost = -score[ys, cols[None, :]] + 0.002 * offs[:, None] ** 2
    n = len(offs)
    acc = cost[:, 0].copy()
    back = np.zeros((n, len(cols)), np.int32)
    pen = smooth * (offs[:, None] - offs[None, :]) ** 2
    for j in range(1, len(cols)):
        tot = acc[None, :] + pen
        back[:, j] = np.argmin(tot, 1)
        acc = tot[np.arange(n), back[:, j]] + cost[:, j]
    path = np.zeros(len(cols), np.int32)
    path[-1] = int(np.argmin(acc))
    for j in range(len(cols) - 1, 0, -1):
        path[j - 1] = back[path[j], j]
    out = y0.copy()
    out[cols] = y0[cols] + offs[path]
    return out


def refine(crop, cp, band_u=12, band_l=9, smooth=0.05):
    h, w = crop.shape[:2]
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab = cv2.GaussianBlur(lab, (0, 0), 0.8)
    yu, xl, xr = curve(cp, UPPER_INNER, w)
    yl, _, _ = curve(cp, LOWER_INNER, w)
    lo, hi = int(max(xl + 2, 0)), int(min(xr - 1, w))
    if hi - lo < 10:
        return yu, yl, lo, hi, None
    p = lip_prob(lab, cp, yu, yl, xl, xr)
    if p is None:
        return yu, yl, lo, hi, None
    above = np.zeros_like(p)
    below = np.zeros_like(p)
    above[2:] = p[:-2]
    below[:-2] = p[2:]
    s_up = above * (1 - below)
    s_lo = (1 - above) * below
    yu2 = snap(s_up, yu, lo, hi, band_u, smooth)
    yl2 = snap(s_lo, yl, lo, hi, band_l, smooth)
    yl2 = np.maximum(yl2, yu2)
    return yu2, yl2, lo, hi, p


def visible(yu, yl, lo, hi, shape, soft=0.8):
    h, w = shape
    gy = np.arange(h, dtype=np.float32)[:, None]
    m = np.clip(gy - yu[None, :] + 0.5, 0, 1) * np.clip(yl[None, :] - gy + 0.5, 0, 1)
    m[:, :lo] = 0
    m[:, hi:] = 0
    return cv2.GaussianBlur(m, (0, 0), soft) if soft > 0 else m
