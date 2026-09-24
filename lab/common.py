import glob, os, pickle
import cv2
import numpy as np
import rigid as R

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
CW, CH = R.CW, R.CH


def keys():
    out = []
    for p in sorted(glob.glob(os.path.join(CACHE, "*_reg.pkl"))):
        out.append(os.path.basename(p)[:-8])
    return out


def load(key):
    d = np.load(os.path.join(CACHE, key + ".npz"), allow_pickle=True)
    regs = pickle.load(open(os.path.join(CACHE, key + "_reg.pkl"), "rb"))
    return d, regs


def warp_region(reg, a, size=(CW, CH), flags=cv2.INTER_LINEAR):
    x0, y0, img = reg
    a = a.copy()
    a[:, 2] -= [x0, y0]
    return cv2.warpAffine(img, a.astype(np.float32), size, flags=flags | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)


def dental(ref=None, idx=None):
    ref = np.load(os.path.join(CACHE, "ref_shape.npy")) if ref is None else ref
    idx = np.load(os.path.join(CACHE, "rigid_idx.npy")) if idx is None else idx
    return R.DentalFrame(ref, idx)
