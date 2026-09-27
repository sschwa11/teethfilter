import json
import sys

import cv2
import numpy as np

sys.path.insert(0, "..")
import dental as D


def head_w(df, pts):
    s, r, t = df.pose(pts)
    ez = np.cross(df.ex, df.ey)
    r2 = (r @ np.stack([df.ex, df.ey], 1))[:2]
    return np.linalg.solve(r2, (r @ ez)[:2])


def collect(key, arch):
    d = dict(np.load(f"cache/{key}.npz", allow_pickle=True))
    cap = cv2.VideoCapture(str(d["path"]))
    af = D.AnchorFill()
    af.reset(json.load(open("../settings.json"))["anchor_offset"])
    af.load_state()
    af.debug = True
    af.upper = arch == "upper"
    af.lower = arch == "lower"
    profs, ws = [], []
    n = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        af.dbg = None
        if d["ok"][n]:
            p = d["pts"][n].astype(np.float64)
            af.process(f, p)
            g = af.dbg
            if g is not None:
                L = cv2.cvtColor(g["crop"], cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
                m = g["mouth"] > 0.5
                if arch == "upper":
                    x0 = af.x_mid
                    band = (np.arange(D.CH)[:, None] >= af.bottom_full[None, :] - 16) & (np.arange(D.CH)[:, None] <= af.bottom_full[None, :] - 2)
                    xs = x0 + np.arange(-8, 112)
                else:
                    if af.lax is None:
                        n += 1
                        continue
                    x0 = af.lax
                    band = (np.arange(D.CH)[:, None] >= af.bottom_full[None, :] + 6) & (np.arange(D.CH)[:, None] <= af.bottom_full[None, :] + 22)
                    xs = x0 - np.arange(-8, 112)
                sel = m & band
                cols = np.clip(np.round(xs).astype(int), 0, D.CW - 1)
                cnt = sel[:, cols].sum(0)
                prof = np.where(cnt >= 5, (L[:, cols] * sel[:, cols]).sum(0) / np.maximum(cnt, 1), np.nan)
                if np.isfinite(prof).mean() > 0.6:
                    profs.append(prof)
                    ws.append(head_w(af.df, p))
        n += 1
    return np.array(profs), np.array(ws)


def shifts(profs, ws, bands, maxs=14):
    fr = np.abs(ws[:, 0]) < 0.03
    ref = np.nanmedian(profs[fr], 0) if fr.sum() >= 10 else np.nanmedian(profs, 0)
    out = []
    for lo, hi in bands:
        ss, ww = [], []
        for pr, w in zip(profs, ws):
            best, bs = -1, None
            seg_idx = np.arange(lo, hi) + 8
            for s in range(-maxs, maxs + 1):
                a = ref[seg_idx]
                bi = seg_idx + s
                if bi.min() < 0 or bi.max() >= pr.size:
                    continue
                b = pr[bi]
                ok = np.isfinite(a) & np.isfinite(b)
                if ok.sum() < 0.7 * a.size:
                    continue
                a2, b2 = a[ok] - a[ok].mean(), b[ok] - b[ok].mean()
                c = float((a2 * b2).sum() / (np.sqrt((a2 ** 2).sum() * (b2 ** 2).sum()) + 1e-6))
                if c > best:
                    best, bs = c, s
            if bs is not None and best > 0.8:
                ss.append(bs)
                ww.append(w[0])
        ss, ww = np.array(ss, float), np.array(ww)
        if len(ss) > 30:
            slope, icpt = np.polyfit(ww, ss, 1)
            out.append((lo, hi, len(ss), slope))
        else:
            out.append((lo, hi, len(ss), np.nan))
    return out


if __name__ == "__main__":
    arch = sys.argv[1]
    keys = sys.argv[2:]
    P, W = [], []
    for k in keys:
        p, w = collect(k, arch)
        P.append(p)
        W.append(w)
        print(k[:19], "frames", len(p), flush=True)
    bands = [(0, 25), (15, 40), (30, 55), (45, 70), (60, 85), (75, 100)]
    for k, p, w in zip(keys, P, W):
        res = shifts(p, w, bands)
        print(k[:19], " ".join(f"[{lo}-{hi}] n{nn} slope {sl:6.1f}" for lo, hi, nn, sl in res), flush=True)
    a = 0.0055
    print("lip-model depth at band centres:", [round(a * ((lo + hi) / 2) ** 2, 1) for lo, hi in bands])
