import sys, cv2, numpy as np
from run_v10 import mouth_box
key, a, b, out, k = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
d = dict(np.load(f"cache/{key}.npz", allow_pickle=True))
ca, cb = cv2.VideoCapture(f"out/{a}/{key}_full.mp4"), cv2.VideoCapture(f"out/{b}/{key}_full.mp4")
cr = cv2.VideoCapture(f"../recordings/{key[:-4]}_raw.mp4")
fr = []; n = 0
while True:
    oa, fa = ca.read(); ob, fb = cb.read(); orr, fw = cr.read()
    if not (oa and ob and orr): break
    if d["ok"][n]:
        x0, y0, w, h = mouth_box(d["pts"][n], fa.shape, 0.8)
        diff = np.abs(fa[y0:y0 + h, x0:x0 + w].astype(int) - fb[y0:y0 + h, x0:x0 + w].astype(int)).mean()
        fr.append((diff, n, [cv2.resize(f[y0:y0 + h, x0:x0 + w], (300, int(300 * h / w)), interpolation=cv2.INTER_CUBIC) for f in (fw, fa, fb)]))
    n += 1
fr.sort(key=lambda t: -t[0])
picked = []
for t in fr:
    if all(abs(t[1] - p[1]) > 10 for p in picked):
        picked.append(t)
    if len(picked) == k: break
rows = []
for diff, n, tiles in sorted(picked, key=lambda t: t[1]):
    row = np.hstack(tiles)
    cv2.putText(row, f"{n} raw | {a} | {b}", (4, 16), 0, 0.5, (0, 255, 255), 1)
    rows.append(row)
cv2.imwrite(out, np.vstack(rows))
