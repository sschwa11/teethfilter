import sys, cv2, numpy as np
sys.path.insert(0, "..")
import dental as D
key = sys.argv[1]
d = dict(np.load(f"cache/{key}.npz", allow_pickle=True))
cap = cv2.VideoCapture(str(d["path"]))
af = D.AnchorFill(); af.debug = True
prev = None; rec = []; n = 0
while True:
    ok, f = cap.read()
    if not ok: break
    af.dbg = None
    if d["ok"][n]: af.process(f, d["pts"][n].astype(np.float64))
    g = af.dbg
    if g is not None and prev is not None:
        dm = np.abs(g["mouth"] - prev["mouth"])
        dau = np.abs(g["alpha_u"] - prev["alpha_u"]); dal = np.abs(g["alpha_l"] - prev["alpha_l"])
        still = dm < 0.1
        comp = g["crop"].astype(np.float32) * (1 - g["alpha"][..., None]) + g["tex"] * g["alpha"][..., None]
        L = comp.mean(2)
        region = (g["alpha"] > 0.3) & (prev["alpha"] > 0.3) & still
        dL = float(np.abs(L - prev["L"])[region].mean()) if region.sum() > 30 else np.nan
        rec.append((n, float(dau[still].sum()), float(dal[still].sum()), dL, float(g["delta"][0] - prev["dx"])))
    prev = None if g is None else {"mouth": g["mouth"], "alpha_u": g["alpha_u"], "alpha_l": g["alpha_l"], "alpha": g["alpha"], "L": (g["crop"].astype(np.float32) * (1 - g["alpha"][..., None]) + g["tex"] * g["alpha"][..., None]).mean(2), "dx": g["delta"][0]}
    n += 1
r = np.array(rec, dtype=float)
print(f"unexplained upper alpha change/frame {np.mean(r[:, 1]):.1f}px  lower {np.mean(r[:, 2]):.1f}px  fill-interior luminance change {np.nanmean(r[:, 3]):.2f}  |ddx| {np.mean(np.abs(r[:, 4])):.2f}")
worst_u = r[np.argsort(-r[:, 1])[:8], 0].astype(int).tolist(); worst_l = r[np.argsort(-r[:, 2])[:8], 0].astype(int).tolist(); worst_L = r[np.argsort(-np.nan_to_num(r[:, 3]))[:8], 0].astype(int).tolist()
print("worst upper", sorted(worst_u)); print("worst lower", sorted(worst_l)); print("worst interior", sorted(worst_L))
