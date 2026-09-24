import json, os, sys, cv2, numpy as np
import common as C
from run_v10 import mouth_box
from make_judge_set import read_frames
a_tag, b_tag, name = sys.argv[1], sys.argv[2], sys.argv[3]
rng = np.random.default_rng(1)
clips = ["2026-09-23_05-47-17_raw", "2026-09-22_23-36-51_raw", "2026-09-23_05-46-00_raw", "2026-09-22_23-02-18", "2026-09-23_05-47-41_raw", "2026-09-22_23-36-22_raw"]
jdir = f"out/judge_{name}"; os.makedirs(jdir, exist_ok=True)
key = []; k = 0
for clip in clips:
    d = dict(np.load(f"cache/{clip}.npz", allow_pickle=True))
    fa = read_frames(f"out/{a_tag}/{clip}_full.mp4", range(len(d["ok"])))
    fb = read_frames(f"out/{b_tag}/{clip}_full.mp4", range(len(d["ok"])))
    diff = [(n, float(np.abs(fa[n].astype(int) - fb[n].astype(int)).mean())) for n in fa if n in fb and d["ok"][n] and n > 60]
    diff = [x for x in diff if x[1] > 0.02]
    if not diff: continue
    pick = [diff[i][0] for i in rng.choice(len(diff), min(4, len(diff)), replace=False)]
    for n in sorted(pick):
        x0, y0, w, h = mouth_box(d["pts"][n], fa[n].shape, scale=1.6)
        tile = lambda img: cv2.resize(img[y0:y0 + h, x0:x0 + w], (420, int(420 * h / w)), interpolation=cv2.INTER_CUBIC)
        first = (a_tag, fa[n]) if rng.random() < 0.5 else (b_tag, fb[n])
        second = (b_tag, fb[n]) if first[0] == a_tag else (a_tag, fa[n])
        img = np.hstack([tile(first[1]), np.full((tile(first[1]).shape[0], 12, 3), 255, np.uint8), tile(second[1])])
        cv2.putText(img, "A", (8, 24), 0, 0.8, (0, 255, 255), 2); cv2.putText(img, "B", (440, 24), 0, 0.8, (0, 255, 255), 2)
        fn = f"pair_{k:03d}.jpg"; cv2.imwrite(f"{jdir}/{fn}", img)
        key.append({"file": fn, "clip": clip, "frame": int(n), "A": first[0], "B": second[0]}); k += 1
json.dump(key, open(f"{jdir}/key.json", "w"), indent=1)
print(jdir, k)
