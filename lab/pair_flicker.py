import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, "..")
import dental as D


def series(key, out_paths):
    d = dict(np.load(f"cache/{key}.npz", allow_pickle=True))
    cap_raw = cv2.VideoCapture(str(d["path"]))
    caps = [cv2.VideoCapture(p) for p in out_paths]
    af = D.AnchorFill()
    af.reset(json.load(open("../settings.json"))["anchor_offset"])
    af.debug = True
    af.upper = False
    af.lower = False
    xm = int(round(af.x_mid))
    rows = np.arange(D.CH)[:, None]
    band = (rows >= af.body_top[None, :] + 2) & (rows <= af.bottom_full[None, :] + 2)
    cols = np.arange(D.CW)
    fz = band & ((cols >= xm - 28) & (cols <= xm - 4))[None, :]
    rz = band & ((cols >= xm + 4) & (cols <= xm + 28))[None, :]
    res = [[] for _ in out_paths]
    n = 0
    while True:
        ok, raw = cap_raw.read()
        outs = [c.read()[1] for c in caps]
        if not ok or any(o is None for o in outs):
            break
        af.dbg = None
        if d["ok"][n]:
            af.process(raw, d["pts"][n].astype(np.float64))
        g = af.dbg
        if g is None or af.locks < 3:
            for r in res:
                r.append((np.nan, np.nan, np.nan, np.nan))
            n += 1
            continue
        a = af.last[0]
        for r, o in zip(res, outs):
            c = cv2.warpAffine(o, a.astype(np.float32), (D.CW, D.CH), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
            L = cv2.cvtColor(c, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
            ref = float(np.percentile(L[rz], 95))
            if ref < 70:
                r.append((np.nan, np.nan, np.nan, np.nan))
                continue
            mz = g["mouth"] > 0.5
            fm, rm = fz & mz, rz & mz
            if fm.sum() < 20 or rm.sum() < 20:
                r.append((np.nan, np.nan, np.nan, np.nan))
                continue
            r.append((float((L[fz] >= 0.62 * ref).mean()), float((L[rz] >= 0.62 * ref).mean()), float(L[fm].mean()), float(L[rm].mean())))
        n += 1
    return [np.array(r) for r in res]


if __name__ == "__main__":
    key = sys.argv[1]
    paths = sys.argv[2:]
    out = series(key, paths)
    for p, s in zip(paths, out):
        lf, lr = s[:, 2], s[:, 3]
        dl = lf - lr
        okl = np.isfinite(dl)
        ddl = np.diff(dl)
        okd = np.isfinite(ddl)
        print(f"  {os.path.basename(os.path.dirname(p))}: fill-minus-real mean L {np.nanmean(dl):+.1f}  |diff| {np.nanmean(np.abs(dl)):.1f}  frame-to-frame |change of diff| {np.nanmean(np.abs(ddl)):.2f}  changes>8 {int((np.abs(ddl[okd]) > 8).sum())} of {int(okd.sum())}  dark-slit frames (real L<60): fill brighter by {np.nanmean(dl[okl & (lr < 60)]):+.1f} (n {int((okl & (lr < 60)).sum())})")
        f, r = s[:, 0], s[:, 1]
        dd = np.diff(f - r)
        ok = np.isfinite(dd) & (r[1:] > 0.25)
        big = np.flatnonzero(ok & (np.abs(dd) > 0.25)) + 1
        print(json.dumps({"video": os.path.basename(os.path.dirname(p)) + "/" + os.path.basename(p), "pairs": int(ok.sum()),
                          "mean |d(fill-real)|": round(float(np.abs(dd[ok]).mean()), 3), "jumps>0.25": int(big.size),
                          "fill/real visible": round(float(np.nanmedian((f / np.maximum(r, 1e-3))[r > 0.25])), 2), "at": big[:30].tolist()}), flush=True)
