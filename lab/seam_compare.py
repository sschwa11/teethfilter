import sys, cv2, numpy as np
sys.path.insert(0, "..")
import dental as D
shifts = [float(v) for v in sys.argv[1].split(",")]
jobs = [("2026-09-24_06-33-17_raw", [50, 125, 150, 175]), ("2026-09-24_06-32-55_raw", [45, 120, 135]), ("2026-09-22_23-02-18", [148, 736])]
rows = []
for key, frames in jobs:
    d = dict(np.load(f"cache/{key}.npz", allow_pickle=True))
    cols = {}
    for sh in shifts:
        cap = cv2.VideoCapture(str(d["path"]))
        af = D.AnchorFill(axis_shift=sh); af.debug = True
        n = 0
        while n <= max(frames):
            ok, f = cap.read()
            if not ok: break
            if d["ok"][n]:
                af.process(f, d["pts"][n].astype(np.float64))
                if n in frames and af.dbg is not None:
                    g = af.dbg
                    comp = np.clip(g["crop"] * (1 - g["alpha"][..., None]) + g["tex"] * g["alpha"][..., None], 0, 255).astype(np.uint8)
                    t = cv2.resize(comp[100:150, 120:200], None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
                    cv2.putText(t, f"{sh:+.2f}", (3, 12), 0, 0.4, (0, 255, 255), 1)
                    cols.setdefault(n, []).append(t)
            n += 1
    for n in frames:
        if n in cols:
            r = np.hstack(cols[n]); cv2.putText(r, f"{key[11:19]} {n}", (3, r.shape[0] - 4), 0, 0.4, (0, 255, 0), 1); rows.append(r)
cv2.imwrite("out/seam_compare.jpg", np.vstack(rows))
