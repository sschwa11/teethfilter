import json
import sys

import cv2
import numpy as np

sys.path.insert(0, "..")
import dental as D

key = sys.argv[1]
opts = {k: eval(v) for k, v in (a.split("=") for a in sys.argv[2:])}
d = dict(np.load(f"cache/{key}.npz", allow_pickle=True))
cap = cv2.VideoCapture(str(d["path"]))
af = D.AnchorFill()
af.reset(json.load(open("../settings.json"))["anchor_offset"])
af.load_state()
af.debug = True
for k, v in opts.items():
    setattr(af, k, v)
recs = []
n = 0
while True:
    ok, f = cap.read()
    if not ok:
        break
    af.dbg = None
    if d["ok"][n]:
        af.process(f, d["pts"][n].astype(np.float64))
        g = af.dbg
        if g is not None:
            xm = int(round(af.x_mid))
            L = cv2.cvtColor(g["crop"], cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
            comp = np.clip(g["crop"] * (1 - g["alpha"][..., None]) + g["tex"] * g["alpha"][..., None], 0, 255).astype(np.uint8)
            Lc = cv2.cvtColor(comp, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
            m = g["mouth"] > 0.5
            rc = slice(xm + 6, xm + 26)
            fc = slice(xm - 26, xm - 6)
            b0 = int(round(af.bottom_full[xm + 16]))
            win = np.zeros(D.CH, bool)
            win[max(b0 - 28, 0):min(b0 + 14, D.CH)] = True
            real_rows = (m[:, rc].mean(1) > 0.8) & win
            if real_rows.sum() >= 6:
                ref = float(np.percentile(L[:, rc][real_rows], 85))
                rb = np.flatnonzero((((L[:, rc] >= 0.6 * ref) & m[:, rc]).mean(1) > 0.6) & win)
                au = g["alpha_u"][:, fc]
                fb = np.flatnonzero(((au > 0.5).mean(1) > 0.6) & win)
                if rb.size and fb.size and ref > 60:
                    lf = float(Lc[fb.min():fb.max() + 1, fc].mean())
                    lr = float(L[rb.min():rb.max() + 1, rc].mean())
                    below_dark = bool(((L[min(rb.max() + 3, D.CH - 1), rc] < 0.5 * ref) & m[min(rb.max() + 3, D.CH - 1), rc]).mean() > 0.6)
                    recs.append((n, rb.max(), fb.max(), rb.max() - fb.max(), lr, lf, af.delta[1], rb.max() - b0, af.w_s[0], af.w_s[1], below_dark))
    n += 1
a = np.array(recs, float)
dd = a[:, 3]
print(f"{key[:19]} {' '.join(sys.argv[2:]) or 'default'}: frames {len(a)}  real-minus-fill bottom (px): median {np.median(dd):.1f}  p10 {np.percentile(dd, 10):.1f}  p90 {np.percentile(dd, 90):.1f}  >=3px {np.mean(dd >= 3):.2f}  frame-to-frame |change| {np.mean(np.abs(np.diff(dd))):.2f}  brightness fill/real {np.median(a[:, 5] / a[:, 4]):.2f}")
np.save(f"out/edge_{key}.npy", a)
ro = a[:, 7]
sel = a[:, 10] > 0.5
if sel.sum() > 20:
    sl, ic = np.polyfit(a[sel, 9], a[sel, 7], 1)
    res = a[sel, 7] - (sl * a[sel, 9] + ic)
    print(f"   edge visible (dark below) frames {int(sel.sum())}: offset vs pitch slope {sl:.1f} px per unit, intercept {ic:.1f}, sd before {a[sel, 7].std():.2f} after {res.std():.2f}, wy range {np.percentile(a[sel, 9], 5):.2f}..{np.percentile(a[sel, 9], 95):.2f}")
print(f"   real tooth bottom minus model edge: median {np.median(ro):.1f}  p10 {np.percentile(ro, 10):.1f}  p90 {np.percentile(ro, 90):.1f}  frame-to-frame |change| {np.mean(np.abs(np.diff(ro))):.2f}   delta_y median {np.median(a[:, 6]):.1f} sd {np.std(a[:, 6]):.2f}")
