import sys, cv2, numpy as np
from run_v10 import mouth_box
key, start, count, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
tags = sys.argv[5:]
d = dict(np.load(f"cache/{key}.npz", allow_pickle=True))
base = key[:-4]
srcs = [("raw", f"../recordings/{base}_raw.mp4"), ("live", f"../recordings/{base}_filtered.mp4")] + [(t, f"out/{t}/{key}_full.mp4") for t in tags]
caps = [cv2.VideoCapture(p) for _, p in srcs]
for c in caps: c.set(cv2.CAP_PROP_POS_FRAMES, start)
p = d["pts"][start + count // 2]
x0, y0, w, h = mouth_box(p, (720, 1280), 0.7)
rows = []
for i in range(count):
    fr = [c.read()[1] for c in caps]
    tiles = [cv2.resize(f[y0:y0 + h, x0:x0 + w], (220, int(220 * h / w)), interpolation=cv2.INTER_CUBIC) for f in fr]
    rows.append(tiles)
grid = np.vstack([np.hstack([rows[i][k] for i in range(count)]) for k in range(len(srcs))])
for k, (name, _) in enumerate(srcs):
    cv2.putText(grid, name, (3, 14 + k * rows[0][0].shape[0]), 0, 0.45, (0, 255, 255), 1)
cv2.imwrite(out, grid)
