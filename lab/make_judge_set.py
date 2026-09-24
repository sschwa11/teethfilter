import json
import os
import sys

import cv2
import numpy as np

import common as C
from run_v10 import mouth_box

OUT = os.path.join(C.HERE, "out")


def read_frames(path, idx):
    cap = cv2.VideoCapture(path)
    out = {}
    n = 0
    want = set(idx)
    while want:
        ok, f = cap.read()
        if not ok:
            break
        if n in want:
            out[n] = f
            want.discard(n)
        n += 1
    return out


def main(tag, per_clip=4, seed=0):
    rng = np.random.default_rng(seed)
    clips = ["2026-09-23_05-47-17_raw", "2026-09-22_23-36-51_raw", "2026-09-23_05-46-00_raw", "2026-09-22_23-02-18", "2026-09-23_05-47-41_raw", "2026-09-22_23-36-22_raw"]
    jdir = os.path.join(OUT, "judge_" + tag)
    os.makedirs(jdir, exist_ok=True)
    key = []
    k = 0
    for clip in clips:
        d = np.load(os.path.join(C.CACHE, clip + ".npz"), allow_pickle=True)
        pts_all = d["pts"]
        opn = np.array([np.linalg.norm(p[13, :2] - p[14, :2]) / max(np.linalg.norm(p[78, :2] - p[308, :2]), 1) if ok else 0 for p, ok in zip(pts_all, d["ok"])])
        cand = np.flatnonzero(opn > 0.1)
        cand = cand[cand > 60]
        if len(cand) < per_clip:
            continue
        pick = sorted(rng.choice(cand, per_clip, replace=False).tolist())
        raw = read_frames(str(d["path"]), pick)
        new = read_frames(os.path.join(OUT, tag, clip + "_full.mp4"), pick)
        v9p = os.path.join(OUT, "v9", clip.replace("_raw", "") + ".mp4")
        old = read_frames(v9p, pick) if os.path.exists(v9p) else {}
        for n in pick:
            pts = pts_all[n].astype(np.float64)
            x0, y0, w, h = mouth_box(pts, raw[n].shape, scale=1.6)
            def tile(img):
                return cv2.resize(img[y0:y0 + h, x0:x0 + w], (420, int(420 * h / w)), interpolation=cv2.INTER_CUBIC)
            pairs = [("raw", raw[n]), (tag, new[n])]
            if n in old:
                pairs.append(("v9", old[n]))
            for other in [p for p in pairs if p[0] != tag]:
                a, b = (other, (tag, new[n])) if rng.random() < 0.5 else ((tag, new[n]), other)
                img = np.hstack([tile(a[1]), np.full((tile(a[1]).shape[0], 12, 3), 255, np.uint8), tile(b[1])])
                cv2.putText(img, "A", (8, 24), 0, 0.8, (0, 255, 255), 2)
                cv2.putText(img, "B", (440, 24), 0, 0.8, (0, 255, 255), 2)
                name = f"pair_{k:03d}.jpg"
                cv2.imwrite(os.path.join(jdir, name), img)
                key.append({"file": name, "clip": clip, "frame": int(n), "A": a[0], "B": b[0]})
                k += 1
    json.dump(key, open(os.path.join(jdir, "key.json"), "w"), indent=1)
    print(jdir, k)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "v10f")
