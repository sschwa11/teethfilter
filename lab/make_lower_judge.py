import json
import os
import sys

import cv2
import numpy as np

from run_v10 import mouth_box

a_tag, b_tag, name = sys.argv[1], sys.argv[2], sys.argv[3]
plan = (("2026-09-25_08-03-41_raw", 8), ("2026-09-25_02-36-08_raw", 8), ("2026-09-25_01-18-25_raw", 6))
rng = np.random.default_rng(7)
jdir = f"out/judge_{name}"
os.makedirs(jdir, exist_ok=True)
key = []
k = 0
for clip, n_seg in plan:
    d = dict(np.load(f"cache/{clip}.npz", allow_pickle=True))
    raw_series = np.load(f"out/lowblink_{clip}.npy")[0]
    vis = np.isfinite(raw_series)
    starts = [s for s in range(20, len(vis) - 10, 4) if vis[s:s + 8].sum() >= 5]
    pick = []
    for s in rng.permutation(starts):
        if all(abs(s - p) >= 16 for p in pick):
            pick.append(int(s))
        if len(pick) == n_seg:
            break
    for s in sorted(pick):
        rows = {}
        for tag in (a_tag, b_tag):
            c = cv2.VideoCapture(f"out/{tag}/{clip}_full.mp4")
            c.set(cv2.CAP_PROP_POS_FRAMES, s)
            x0, y0, w, h = mouth_box(d["pts"][s + 4], (720, 1280), 0.75)
            tiles = [cv2.resize(c.read()[1][y0:y0 + h, x0:x0 + w], (240, int(240 * h / w)), interpolation=cv2.INTER_CUBIC) for _ in range(8)]
            rows[tag] = np.hstack(tiles)
        order = [a_tag, b_tag] if rng.random() < 0.5 else [b_tag, a_tag]
        img = np.vstack([rows[order[0]], np.full((10, rows[a_tag].shape[1], 3), 255, np.uint8), rows[order[1]]])
        cv2.putText(img, "A", (4, 20), 0, 0.7, (0, 255, 255), 2)
        cv2.putText(img, "B", (4, rows[a_tag].shape[0] + 30), 0, 0.7, (0, 255, 255), 2)
        fn = f"pair_{k:03d}.jpg"
        cv2.imwrite(f"{jdir}/{fn}", img)
        key.append({"file": fn, "clip": clip, "start": s, "A": order[0], "B": order[1]})
        k += 1
json.dump(key, open(f"{jdir}/key.json", "w"), indent=1)
print(jdir, k)
