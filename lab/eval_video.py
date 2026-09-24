import json
import os
import sys

import cv2
import numpy as np

import common as C
import edge_track as E
import render_v10c as V


def evaluate(key, out_path, mouth_src, model=None, max_frames=None):
    model = model or V.Model()
    d = np.load(os.path.join(C.CACHE, key + ".npz"), allow_pickle=True)
    cap_raw = cv2.VideoCapture(str(d["path"]))
    cap_out = cv2.VideoCapture(out_path) if out_path else None
    df = C.dental()
    et = E.EdgeTracker(model.x_mid + model.edge_offset, float(model.bottom[int(model.x_mid) + 18]))
    slot = model.slot > 0.5
    donor_m = model.donor_mask > 0.5
    stats = []
    prev = None
    n = 0
    delta = et.off.copy()
    while True:
        ok, raw = cap_raw.read()
        if not ok or (max_frames and n >= max_frames):
            break
        out = cap_out.read()[1] if cap_out is not None else raw
        if out is None:
            break
        if d["ok"][n]:
            pts = d["pts"][n].astype(np.float64)
            a0 = df.affine(df.pose(pts))
            flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
            c_raw = cv2.warpAffine(raw, a0.astype(np.float32), (C.CW, C.CH), flags=flags, borderMode=cv2.BORDER_REPLICATE)
            mouth0 = mouth_src(raw, pts, n)
            m = et.measure(c_raw, mouth0)
            good = m is not None and m["conf"] >= 0.12
            et.update(m if good else None)
            if good:
                delta = np.array([m["x"] - et.x0, (m["y"] - et.b0) if m["y"] is not None else et.off[1]])
            a = a0.copy()
            a[:, 2] += a0[:, :2] @ delta
            c_out = cv2.warpAffine(out, a.astype(np.float32), (C.CW, C.CH), flags=flags, borderMode=cv2.BORDER_REPLICATE)
            c_r = cv2.warpAffine(raw, a.astype(np.float32), (C.CW, C.CH), flags=flags, borderMode=cv2.BORDER_REPLICATE)
            mouth = V.shift_img(mouth0, delta, cv2.BORDER_CONSTANT, 0)
            L = cv2.cvtColor(c_out, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
            Lr = cv2.cvtColor(c_r, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
            vis = slot & (mouth > 0.6)
            donor = donor_m & (mouth > 0.6)
            s = {}
            if vis.sum() > 30 and donor.sum() > 30:
                tl = float(np.percentile(Lr[donor], 60))
                s["leak_raw"] = float((Lr[vis] < 0.6 * tl).mean())
                s["leak"] = float((L[vis] < 0.6 * tl).mean())
                s["changed"] = float((np.abs(L - Lr)[vis] > 12).mean())
                s["lip_touch"] = float((np.abs(L - Lr)[slot & (mouth < 0.2)] > 12).mean())
                if prev is not None:
                    pv, pd, pL, pLr = prev
                    both = vis & pv
                    dd = donor & pd
                    if both.sum() > 30 and dd.sum() > 30:
                        s["flick_fill"] = float(np.abs(L - pL)[both].mean())
                        s["flick_real"] = float(np.abs(Lr - pLr)[dd].mean())
                prev = (vis, donor, L, Lr)
            else:
                prev = None
            stats.append(s)
        else:
            prev = None
        n += 1
    f = lambda k: [s[k] for s in stats if k in s]
    res = {k: float(np.mean(f(k))) if f(k) else float("nan") for k in ("leak_raw", "leak", "changed", "lip_touch", "flick_fill", "flick_real")}
    res["vis_frames"] = len(f("leak"))
    res["flick_ratio"] = res["flick_fill"] / res["flick_real"] if res["flick_real"] > 0 else float("nan")
    return res


if __name__ == "__main__":
    from run_v10c import MouthSource
    key, src = sys.argv[1], sys.argv[2]
    ms = MouthSource(src)
    ms.set_video(key)
    for path in sys.argv[3:]:
        ms.last = None
        r = evaluate(key, None if path == "raw" else path, ms)
        print(path, json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}), flush=True)
