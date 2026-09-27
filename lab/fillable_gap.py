import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, "..")
import dental as D


def series(key, out_paths, zones):
    d = dict(np.load(f"cache/{key}.npz", allow_pickle=True))
    cap_raw = cv2.VideoCapture(str(d["path"]))
    caps = [cv2.VideoCapture(p) for p in out_paths]
    af = D.AnchorFill()
    af.reset(json.load(open("../settings.json"))["anchor_offset"])
    af.debug = True
    af.upper = False
    af.lower = True
    rows = np.arange(D.CH)[:, None]
    xs = np.arange(D.CW).astype(np.float64)
    res = {z: [[] for _ in out_paths] for z in zones}
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
        if g is None or af.locks < 3 or af.w_s is None:
            for z in zones:
                for r in res[z]:
                    r.append(np.nan)
            n += 1
            continue
        a, mouth = af.last[0], g["mouth"]
        Lr = cv2.cvtColor(g["crop"], cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
        inm = mouth > 0.7
        donor = af.donor_mask & inm
        ref = float(np.percentile(Lr[donor], 85)) if donor.sum() >= 30 else 0.0
        lb = np.where(inm.any(0), D.CH - 1 - np.argmax(inm[::-1], 0), -1)
        wx = float(af.w_s[0])
        masks = {}
        for z, (lo, hi) in zones.items():
            if z.startswith("upper"):
                axis = af.x_mid
                zr = (rows >= 88) & (rows <= af.bottom_full[None, :] - 2)
            else:
                if af.lax is None:
                    masks[z] = None
                    continue
                axis = af.lax
                zr = (rows > af.bottom_full[None, :] + 3) & (rows < lb[None, :] - 1) & (rows >= lb[None, :] - 22)
            X = af.arch_solve(xs - axis, af.arch_a, wx)
            ud = np.clip(np.round(xs - 2 * X).astype(int), 0, D.CW - 1)
            rel = xs - axis
            zc = (rel >= lo) & (rel < hi)
            donor_bright = (Lr[:, ud] >= 0.6 * ref) & (mouth[:, ud] > 0.7)
            masks[z] = inm & zr & zc[None, :] & (Lr < 0.5 * ref) & donor_bright
        outs_c = []
        for o in outs:
            c = cv2.warpAffine(o, a.astype(np.float32), (D.CW, D.CH), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
            outs_c.append(cv2.cvtColor(c, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32))
        for z in zones:
            m = masks.get(z)
            ok_z = m is not None and ref >= 60 and m.sum() >= 25
            for r, L in zip(res[z], outs_c):
                r.append(float((L[m] < 0.5 * ref).mean()) if ok_z else np.nan)
        n += 1
    return res


if __name__ == "__main__":
    key = sys.argv[1]
    paths = sys.argv[2:]
    zones = {"upper_front": (-66, -2), "upper_outer": (-92, -60), "lower_front": (2, 76), "lower_outer": (70, 105)}
    res = series(key, paths, zones)
    for i, p in enumerate(paths):
        parts = []
        for z in zones:
            s = np.array(res[z][i])
            fin = np.isfinite(s)
            parts.append(f"{z}: n{int(fin.sum())} left {s[fin].mean():.3f} >0.3 {(s[fin] > 0.3).mean():.3f}" if fin.any() else f"{z}: none")
        print(f"{os.path.basename(os.path.dirname(p)):12s} " + " | ".join(parts), flush=True)
