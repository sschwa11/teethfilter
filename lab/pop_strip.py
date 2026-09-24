import sys, cv2, numpy as np
sys.path.insert(0, "..")
import dental as D
key = sys.argv[1]; centers = [int(v) for v in sys.argv[2].split(",")]; out = sys.argv[3]
d = dict(np.load(f"cache/{key}.npz", allow_pickle=True))
cap = cv2.VideoCapture(str(d["path"]))
af = D.AnchorFill(); af.debug = True
for k, v in (a.split("=") for a in sys.argv[4:]):
    setattr(af, k, eval(v))
want = {}
for c in centers:
    for k in range(c - 3, c + 4): want[k] = c
rows = {c: [] for c in centers}
n = 0
while n <= max(want):
    ok, f = cap.read()
    if not ok: break
    af.dbg = None
    if d["ok"][n]: af.process(f, d["pts"][n].astype(np.float64))
    g = af.dbg
    if n in want and g is not None:
        comp = np.clip(g["crop"] * (1 - g["alpha"][..., None]) + g["tex"] * g["alpha"][..., None], 0, 255).astype(np.uint8)
        m = cv2.cvtColor((g["mouth"] * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        al = cv2.cvtColor((g["alpha"] * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        t = np.vstack([g["crop"][95:165, 70:250], comp[95:165, 70:250], m[95:165, 70:250], al[95:165, 70:250]])
        t = cv2.resize(t, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_NEAREST)
        cv2.putText(t, str(n), (3, 14), 0, 0.45, (0, 255, 255), 1)
        rows[want[n]].append(t)
    n += 1
img = np.vstack([np.hstack(r) for r in rows.values() if len(r) == 7])
cv2.imwrite(out, img)
