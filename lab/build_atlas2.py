import glob
import os
import sys

import cv2
import numpy as np

import common as C
from render_v10 import highpass
from render_v10c import shift_img

CW, CH = C.CW, C.CH
TPL_BOX = (112, 144, 152, 236)
REF_BOX = (120, 134, 164, 190)


def load_teach(key):
    z = np.load(os.path.join(C.CACHE, key + "_teach.npz"))
    lab = z["labels"][:, :, 0].astype(np.float32) / 255
    mouth = 0.6 * lab[:, 0] + 0.4 * lab[:, 1]
    return z["crops"], mouth, z["ok"]


def track(crops, ok, tpl, pad=22):
    y0, y1, x0, x1 = TPL_BOX
    res = np.full((len(crops), 3), np.nan)
    for i in np.flatnonzero(ok):
        g = highpass(cv2.cvtColor(crops[i], cv2.COLOR_BGR2GRAY).astype(np.float32))
        r = cv2.matchTemplate(g[y0 - pad:y1 + pad, x0 - pad:x1 + pad], tpl, cv2.TM_CCOEFF_NORMED)
        _, mv, _, ml = cv2.minMaxLoc(r)
        px, py = ml
        dx = dy = 0.0
        if 0 < px < r.shape[1] - 1:
            den = r[py, px - 1] - 2 * r[py, px] + r[py, px + 1]
            dx = 0.5 * (r[py, px - 1] - r[py, px + 1]) / den if abs(den) > 1e-6 else 0.0
        if 0 < py < r.shape[0] - 1:
            den = r[py - 1, px] - 2 * r[py, px] + r[py + 1, px]
            dy = 0.5 * (r[py - 1, px] - r[py + 1, px]) / den if abs(den) > 1e-6 else 0.0
        res[i] = (px + dx - pad, py + dy - pad, mv)
    return res


def template_from(crop):
    y0, y1, x0, x1 = TPL_BOX
    return highpass(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32))[y0:y1, x0:x1].copy()


def main(keys):
    z = np.load(os.path.join(C.CACHE, "model_v10.npz"))
    tpl0 = z["tpl"].astype(np.float32)
    acc = np.zeros((CH, CW, 3))
    wsum = np.zeros((CH, CW))
    vis = np.zeros((CH, CW))
    refs = []
    per = {}
    for key in keys:
        crops, mouth, ok = load_teach(key)
        res = track(crops, ok, tpl0)
        good = res[:, 2] > 0.6
        if good.sum() >= 8:
            best = np.flatnonzero(good)[np.argsort(-res[good, 2])[:8]]
            stack = [shift_img(crops[i].astype(np.float32), res[i, :2]) for i in best]
            tpl1 = template_from(np.median(stack, 0).astype(np.uint8))
            res2 = track(crops, ok, tpl1)
            off = np.median(res[best, :2], 0)
            res2[:, :2] += off
            res = res2
            good = res[:, 2] > 0.6
        if good.sum() >= 5:
            med = np.nanmedian(res[good, :2], 0)
            good &= np.all(np.abs(res[:, :2] - med) < 10, 1)
        per[key] = (res, good)
        used = 0
        y0, y1, x0, x1 = REF_BOX
        for i in np.flatnonzero(good):
            crop = shift_img(crops[i].astype(np.float32), res[i, :2])
            m = shift_img(mouth[i], res[i, :2], cv2.BORDER_CONSTANT, 0)
            core = m[y0:y1, x0:x1] > 0.8
            if core.mean() < 0.6:
                continue
            g = cv2.cvtColor(np.clip(crop, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
            sharp = float(cv2.Laplacian(g[y0 - 4:y1 + 4, x0 - 8:x1 + 8], cv2.CV_32F).var())
            ref = np.array([np.median(crop[y0:y1, x0:x1, c][core]) for c in range(3)])
            if ref.min() < 30:
                continue
            w = np.clip((m - 0.7) / 0.3, 0, 1) * min(sharp, 400.0)
            acc += crop / ref[None, None, :] * w[..., None]
            wsum += w
            vis += (m > 0.7)
            refs.append(ref)
            used += 1
        print(f"{key}: tracked {int(good.sum())}/{int(ok.sum())} used {used}", flush=True)
    ratio = acc / np.maximum(wsum, 1e-6)[..., None]
    ref = np.median(np.array(refs), 0)
    atlas = np.clip(ratio * ref[None, None, :], 0, 255).astype(np.float32)
    np.savez_compressed(os.path.join(C.CACHE, "atlas_v2.npz"), atlas=atlas, wsum=wsum, vis=vis, ref=ref)
    img = atlas.astype(np.uint8).copy()
    img[wsum < 1] = 0
    cv2.imwrite(os.path.join(C.HERE, "out", "atlas_v2.jpg"), cv2.resize(img[80:180, 40:290], None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC))
    return per


if __name__ == "__main__":
    keys = sys.argv[1:] or [os.path.basename(p)[:-10] for p in sorted(glob.glob(os.path.join(C.CACHE, "*_teach.npz")))]
    main(keys)
