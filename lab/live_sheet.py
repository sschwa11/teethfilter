import sys, cv2, numpy as np
from run_v10 import mouth_box
key = sys.argv[1]; every = int(sys.argv[2]); out = sys.argv[3]
d = dict(np.load(f"cache/{key}.npz", allow_pickle=True))
base = key[:-4]
ca = cv2.VideoCapture(f"../recordings/{base}_raw.mp4"); cb = cv2.VideoCapture(f"../recordings/{base}_filtered.mp4")
tiles = []; n = 0
while True:
    oa, fa = ca.read(); ob, fb = cb.read()
    if not oa or not ob: break
    if n % every == 0 and d["ok"][n]:
        x0, y0, w, h = mouth_box(d["pts"][n], fa.shape, 1.0)
        th = int(360 * 0.75 / 1.3)
        t = np.hstack([cv2.resize(fa[y0:y0+h, x0:x0+w], (360, th), interpolation=cv2.INTER_CUBIC), cv2.resize(fb[y0:y0+h, x0:x0+w], (360, th), interpolation=cv2.INTER_CUBIC)])
        cv2.putText(t, str(n), (4, 16), 0, 0.5, (0, 255, 0), 1)
        tiles.append(t)
    n += 1
rows = [np.hstack(tiles[i:i+2]) for i in range(0, len(tiles) - 1, 2)]
for part in range(0, len(rows), 8):
    cv2.imwrite(out.replace(".jpg", f"_{part // 8}.jpg"), np.vstack(rows[part:part + 8]))
print(n, len(tiles))
