import glob
import os
import sys

import cv2
import numpy as np

import build_atlas2 as B2
import common as C
import edge_track as E
from render_v10c import shift_img

CW, CH = C.CW, C.CH
X_E, B_E = 160.0, 137.0
SKIP = ("2026-09-22_18-36-31", "2026-09-22_18-54-22")


def align_session(crops, mouth, ok):
    t = E.EdgeTracker(X_E, B_E)
    meas = {}
    for p in range(2):
        for i in np.flatnonzero(ok):
            m = t.measure(crops[i], mouth[i])
            good = m is not None and m["conf"] >= 0.12
            t.update(m if good else None)
            if p == 1 and good:
                meas[i] = m
    deltas = {}
    for i, m in meas.items():
        dy = (m["y"] - B_E) if m["y"] is not None else t.off[1]
        if m["y"] is not None and abs(dy - t.off[1]) > 8:
            dy = t.off[1]
        deltas[i] = np.array([m["x"] - X_E, dy])
    return deltas, t.off.copy()


def main(keys):
    acc = np.zeros((CH, CW, 3))
    wsum = np.zeros((CH, CW))
    tooth = np.zeros((CH, CW))
    vis = np.zeros((CH, CW))
    refs = []
    for key in keys:
        crops, mouth, ok = B2.load_teach(key)
        deltas, off = align_session(crops, mouth, ok)
        used = 0
        for i, d in deltas.items():
            crop = shift_img(crops[i].astype(np.float32), d)
            m = shift_img(mouth[i], d, cv2.BORDER_CONSTANT, 0)
            L = cv2.cvtColor(np.clip(crop, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB)[..., 0]
            inm = m > 0.7
            if inm.sum() < 100:
                continue
            thr, _ = cv2.threshold(L[inm].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            br = inm & (L > thr)
            core = br[122:134, 164:190]
            if core.mean() < 0.5:
                continue
            ref = np.array([np.median(crop[122:134, 164:190, c][core]) for c in range(3)])
            if ref.min() < 30:
                continue
            g = L.astype(np.float32)
            sharp = float(cv2.Laplacian(g[114:142, 156:196], cv2.CV_32F).var())
            w = np.clip((m - 0.7) / 0.3, 0, 1) * min(sharp, 400.0)
            acc += crop / ref[None, None, :] * w[..., None]
            wsum += w
            tooth += br
            vis += inm
            refs.append(ref)
            used += 1
        print(f"{key}: aligned {len(deltas)}/{int(ok.sum())} used {used} session offset {off.round(1)}", flush=True)
    ratio = acc / np.maximum(wsum, 1e-6)[..., None]
    ref = np.median(np.array(refs), 0)
    atlas = np.clip(ratio * ref[None, None, :], 0, 255).astype(np.float32)
    prob = tooth / np.maximum(vis, 1)
    np.savez_compressed(os.path.join(C.CACHE, "atlas_v3.npz"), atlas=atlas, wsum=wsum, vis=vis, prob=prob, ref=ref)
    img = atlas.astype(np.uint8).copy()
    img[wsum < 1] = 0
    pv = cv2.applyColorMap((np.clip(prob, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    pv[vis < 5] = 0
    out = np.vstack([img[80:180, 40:290], pv[80:180, 40:290]])
    cv2.imwrite(os.path.join(C.HERE, "out", "atlas_v3.jpg"), cv2.resize(out, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC))


if __name__ == "__main__":
    keys = sys.argv[1:] or [os.path.basename(p)[:-10] for p in sorted(glob.glob(os.path.join(C.CACHE, "*_teach.npz")))]
    main([k for k in keys if k not in SKIP])
