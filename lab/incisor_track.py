import sys, pickle, cv2, numpy as np
key, tf, y0, y1, x0, x1 = sys.argv[1], int(sys.argv[2]), *map(int, sys.argv[3:7])
crops = pickle.load(open(f"cache/{key}_crops.pkl", "rb"))
g = lambda c: cv2.cvtColor(c, cv2.COLOR_BGR2GRAY).astype(np.float32)
tpl = g(crops[tf])[y0:y1, x0:x1]
cv2.imwrite(f"cache/{key}_tpl.png", crops[tf][y0 - 20:y1 + 20, x0 - 20:x1 + 20])
pad = 25
res = []
for i, c in enumerate(crops):
    if c is None:
        res.append((np.nan, np.nan, 0)); continue
    win = g(c)[y0 - pad:y1 + pad, x0 - pad:x1 + pad]
    r = cv2.matchTemplate(win, tpl, cv2.TM_CCOEFF_NORMED)
    _, mv, _, ml = cv2.minMaxLoc(r)
    px, py = ml
    if 0 < px < r.shape[1] - 1 and 0 < py < r.shape[0] - 1:
        dx = 0.5 * (r[py, px - 1] - r[py, px + 1]) / (r[py, px - 1] - 2 * r[py, px] + r[py, px + 1] + 1e-9)
        dy = 0.5 * (r[py - 1, px] - r[py + 1, px]) / (r[py - 1, px] - 2 * r[py, px] + r[py + 1, px] + 1e-9)
    else:
        dx = dy = 0
    res.append((px + dx - pad, py + dy - pad, mv))
res = np.array(res)
good = res[:, 2] > 0.8
print(f"{key}: matched {good.sum()}/{len(res)}")
p = res[good, :2]
print("offset mean", p.mean(0).round(2), "std", p.std(0).round(2), "p5-p95 x", np.percentile(p[:, 0], [5, 95]).round(1), "y", np.percentile(p[:, 1], [5, 95]).round(1))
idx = np.flatnonzero(good)
cons = idx[1:][np.diff(idx) == 1]
d = res[cons, :2] - res[cons - 1, :2]
print("frame-to-frame |d| median", np.median(np.abs(d), 0).round(2), "p90", np.percentile(np.abs(d), 90, axis=0).round(2))
np.save(f"cache/{key}_inc.npy", res)
