import os
import sys

import cv2
import numpy as np

import common as C

sys.path.insert(0, os.path.dirname(C.HERE))
import teeth_filter as T

CW, CH = C.CW, C.CH
GY = np.mgrid[0:CH, 0:CW][0].astype(np.float32)


def upward(mask, px):
    out = mask.copy()
    for k in range(1, px + 1):
        out[:-k] |= mask[k:]
    return out


def mouth_masks(cp):
    mk = np.zeros((CH, CW), np.uint8)
    cv2.fillPoly(mk, [np.round(cp[T.INNER_POLY]).astype(np.int32)], 255, lineType=cv2.LINE_AA)
    inside = cv2.erode(mk, np.ones((3, 3), np.uint8)) > 127
    return inside, upward(inside, 10)


def lip_lines(cp):
    up = cp[T.UPPER_INNER]
    lo = cp[T.LOWER_INNER]
    xs = np.arange(CW)
    ou, ol = np.argsort(up[:, 0]), np.argsort(lo[:, 0])
    return np.interp(xs, up[ou, 0], up[ou, 1]).astype(np.float32), np.interp(xs, lo[ol, 0], lo[ol, 1]).astype(np.float32)


def teeth_mask(crop, cp):
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    inside, inside_up = mouth_masks(cp)
    t = T.otsu_split(lab[..., 0][inside])
    if t is None:
        return None, inside_up, lab
    yu, yl = lip_lines(cp)
    bright = T.lip_free_bright(lab, inside_up, t, yu, yl)
    return bright, inside_up, lab


class Tracker:
    def __init__(self, tpl_gray, box, pad=24):
        self.tpl = tpl_gray
        self.box = box
        self.pad = pad

    def match(self, crop):
        y0, y1, x0, x1 = self.box
        p = self.pad
        g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)[y0 - p:y1 + p, x0 - p:x1 + p]
        r = cv2.matchTemplate(g, self.tpl, cv2.TM_CCOEFF_NORMED)
        _, mv, _, ml = cv2.minMaxLoc(r)
        px, py = ml
        dx = dy = 0.0
        if 0 < px < r.shape[1] - 1:
            den = r[py, px - 1] - 2 * r[py, px] + r[py, px + 1]
            dx = 0.5 * (r[py, px - 1] - r[py, px + 1]) / den if abs(den) > 1e-6 else 0.0
        if 0 < py < r.shape[0] - 1:
            den = r[py - 1, px] - 2 * r[py, px] + r[py + 1, px]
            dy = 0.5 * (r[py - 1, px] - r[py + 1, px]) / den if abs(den) > 1e-6 else 0.0
        return px + dx - p, py + dy - p, mv


def shifted(a, dx, dy):
    b = a.copy()
    b[:, 2] += a[:, :2] @ np.array([dx, dy])
    return b
