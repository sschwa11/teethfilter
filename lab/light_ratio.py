import sys

import cv2
import numpy as np

sys.path.insert(0, "..")
import dental as D

key = sys.argv[1]
d = dict(np.load(f"cache/{key}.npz", allow_pickle=True))
cap = cv2.VideoCapture(str(d["path"]))
af = D.AnchorFill()
af.debug = True
af.upper = False
af.lower = False
xm = int(af.x_mid)
rows = np.arange(D.CH)[:, None]
out = []
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
        inm = g["mouth"] > 0.5
        if inm[:, xm - 50:xm + 50].sum() > 100:
            L = cv2.cvtColor(g["crop"], cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
            lb = np.where(inm.any(0), D.CH - 1 - np.argmax(inm[::-1], 0), -1)
            ub = np.where(inm.any(0), np.argmax(inm, 0), -1)
            lower = (rows >= lb[None, :] + 3) & (rows <= lb[None, :] + 12) & (lb >= 0)[None, :]
            upper = (rows <= ub[None, :] - 3) & (rows >= ub[None, :] - 10) & (ub >= 0)[None, :]
            vals = []
            band = inm & (rows > af.bottom_full[None, :] + 2)
            for (l0, l1) in ((58, 82),):
                lm = band[:, xm - l1:xm - l0]
                rm = band[:, xm + l0:xm + l1][:, ::-1]
                Ll = L[:, xm - l1:xm - l0]
                Lr = L[:, xm + l0:xm + l1][:, ::-1]
                if lm.sum() > 40 and rm.sum() > 40:
                    vals.append(float(np.percentile(Lr[rm], 85) / max(np.percentile(Ll[lm], 85), 1)))
                else:
                    vals.append(np.nan)
            ub_ = np.where(inm.any(0), np.argmax(inm, 0), -1)
            up = inm & (rows < af.bottom_full[None, :]) & (rows > ub_[None, :])
            lm = up[:, xm - 90:xm - 68]; rm = up[:, xm + 68:xm + 90][:, ::-1]
            if lm.sum() > 30 and rm.sum() > 30:
                vals.append(float(np.percentile(L[:, xm + 68:xm + 90][:, ::-1][rm], 85) / max(np.percentile(L[:, xm - 90:xm - 68][lm], 85), 1)))
            else:
                vals.append(np.nan)
            out.append((n, *vals))
    n += 1
a = np.array(out)
print(key[:19], "frames", len(a), "lower canines R/L median %.3f (p10 %.3f p90 %.3f)" % tuple(np.nanpercentile(a[:, 1], [50, 10, 90])),
      "upper canines R/L median %.3f n %d" % (np.nanmedian(a[:, 2]), np.isfinite(a[:, 2]).sum()), "frame-to-frame sd %.3f" % np.nanstd(np.diff(a[:, 1])))
