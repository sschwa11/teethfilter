import json
import os
import sys

import cv2
import numpy as np

from run_v10 import mouth_box

key, out_video, name = sys.argv[1], sys.argv[2], sys.argv[3]
step = int(sys.argv[4]) if len(sys.argv) > 4 else 20
d = dict(np.load(f"cache/{key}.npz", allow_pickle=True))
opn = np.array([np.linalg.norm(p[13, :2] - p[14, :2]) / max(np.linalg.norm(p[78, :2] - p[308, :2]), 1) if ok else 0 for p, ok in zip(d["pts"], d["ok"])])
jdir = f"out/sweep_{name}"
os.makedirs(jdir, exist_ok=True)
n_frames = len(d["ok"])
starts = []
s = 0
while s < n_frames - 8:
    if (opn[s:s + 8] > 0.06).sum() >= 4 and d["ok"][s:s + 8].all():
        starts.append(s)
        s += step
    else:
        s += 2
need = {f: s for s in starts for f in range(s, s + 8)}
boxes = {s: mouth_box(d["pts"][s + 4], (720, 1280), 0.8) for s in starts}
crops = {}
cr = cv2.VideoCapture(f"../recordings/{key[:-4]}_raw.mp4")
co = cv2.VideoCapture(out_video)
n = 0
while n <= max(need):
    a, fa = cr.read()
    b, fb = co.read()
    if not a or not b:
        break
    if n in need:
        x0, y0, w, h = boxes[need[n]]
        crops[n] = (fa[y0:y0 + h, x0:x0 + w].copy(), fb[y0:y0 + h, x0:x0 + w].copy())
    n += 1
index = []
for k, s in enumerate(starts):
    x0, y0, w, h = boxes[s]
    tw = 200
    th = int(tw * h / w)
    rows = []
    for which in (0, 1):
        tiles = []
        for i in range(8):
            t = cv2.resize(crops[s + i][which], (tw, th), interpolation=cv2.INTER_CUBIC)
            tiles.append(t)
        rows.append(np.hstack(tiles))
    img = np.vstack([rows[0], np.full((8, rows[0].shape[1], 3), 255, np.uint8), rows[1]])
    for i in range(8):
        cv2.putText(img, str(s + i), (i * tw + 3, 14), 0, 0.45, (0, 255, 255), 1)
    cv2.putText(img, "RAW", (3, th - 6), 0, 0.5, (0, 255, 255), 1)
    cv2.putText(img, "FILTERED", (3, 2 * th + 2), 0, 0.5, (0, 255, 255), 1)
    fn = f"strip_{k:03d}.jpg"
    cv2.imwrite(f"{jdir}/{fn}", img)
    index.append({"file": fn, "start": s})
json.dump(index, open(f"{jdir}/index.json", "w"), indent=1)
print(jdir, len(index))
