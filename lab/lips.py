import cv2
import numpy as np

UPPER_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
LOWER_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]


def curve(cp, ids, width):
    p = cp[ids]
    o = np.argsort(p[:, 0])
    xs = np.arange(width)
    return np.interp(xs, p[o, 0], p[o, 1]).astype(np.float32), float(p[:, 0].min()), float(p[:, 0].max())


def features(crop):
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab = cv2.GaussianBlur(lab, (0, 0), 1.0)
    return lab[..., 0], lab[..., 1]


def snap(score, y0, lo, hi, band, smooth):
    h, w = score.shape
    offs = np.arange(-band, band + 1)
    cols = np.arange(lo, hi)
    ys = np.clip(np.round(y0[cols][None, :] + offs[:, None]).astype(int), 0, h - 1)
    cost = -score[ys, cols[None, :]]
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


def refine(crop, cp, band=10, smooth=0.6):
    h, w = crop.shape[:2]
    L, A = features(crop)
    yu, xl, xr = curve(cp, UPPER_INNER, w)
    yl, _, _ = curve(cp, LOWER_INNER, w)
    lip_a = np.median(A[np.clip(np.round(yl + 8).astype(int), 0, h - 1), np.arange(w)][int(xl) + 10:int(xr) - 10]) if xr - xl > 30 else 150.0
    gy_l = np.zeros_like(L)
    gy_l[1:-1] = (L[2:] - L[:-2]) / 2
    gy_a = np.zeros_like(A)
    gy_a[1:-1] = (A[2:] - A[:-2]) / 2
    s_up = np.abs(gy_l) * 0.5 + np.clip(-gy_a, 0, None) * 2.0
    s_lo = np.abs(gy_l) * 0.5 + np.clip(gy_a, 0, None) * 2.0
    lo, hi = int(max(xl + 2, 0)), int(min(xr - 1, w))
    if hi - lo < 10:
        return yu, yl, lo, hi
    yu2 = snap(s_up, yu, lo, hi, band, smooth)
    yl2 = snap(s_lo, yl, lo, hi, band, smooth)
    yl2 = np.maximum(yl2, yu2)
    return yu2, yl2, lo, hi


def opening_mask(yu, yl, lo, hi, shape, soft=1.0):
    h, w = shape
    gy = np.arange(h, dtype=np.float32)[:, None]
    m = np.clip(gy - yu[None, :] + 0.5, 0, 1) * np.clip(yl[None, :] - gy + 0.5, 0, 1)
    m[:, :lo] = 0
    m[:, hi:] = 0
    if soft > 0:
        m = cv2.GaussianBlur(m, (0, 0), soft)
    return m
