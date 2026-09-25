import json, os, sys, cv2, numpy as np
from run_v10 import mouth_box
new_tag, name = sys.argv[1], sys.argv[2]
base = sys.argv[3] if len(sys.argv) > 3 else "live"
base_path = (lambda b: f"../recordings/{b}_filtered.mp4") if base == "live" else (lambda b: f"out/{base}/{b}_raw_full.mp4")
rng = np.random.default_rng(3)
jdir = f"out/judge_{name}"; os.makedirs(jdir, exist_ok=True)
key = []; k = 0
for clip, n_seg in (("2026-09-24_06-33-17_raw", 8), ("2026-09-24_06-32-55_raw", 5), ("2026-09-25_00-43-21_raw", 7)):
    d = dict(np.load(f"cache/{clip}.npz", allow_pickle=True))
    stem = clip[:-4]
    opn = np.array([np.linalg.norm(p[13, :2] - p[14, :2]) / max(np.linalg.norm(p[78, :2] - p[308, :2]), 1) for p in d["pts"]])
    starts = [s for s in range(20, len(opn) - 10, 4) if (opn[s:s + 8] > 0.06).sum() >= 5]
    pick = sorted(rng.choice(starts, n_seg, replace=False).tolist())
    for s in pick:
        rows = {}
        for tag, path in (("live", base_path(stem)), (new_tag, f"out/{new_tag}/{clip}_full.mp4")):
            c = cv2.VideoCapture(path); c.set(cv2.CAP_PROP_POS_FRAMES, s)
            x0, y0, w, h = mouth_box(d["pts"][s + 4], (720, 1280), 0.9)
            tiles = [cv2.resize(c.read()[1][y0:y0 + h, x0:x0 + w], (210, int(210 * h / w)), interpolation=cv2.INTER_CUBIC) for _ in range(8)]
            rows[tag] = np.hstack(tiles)
        order = ["live", new_tag] if rng.random() < 0.5 else [new_tag, "live"]
        img = np.vstack([rows[order[0]], np.full((10, rows["live"].shape[1], 3), 255, np.uint8), rows[order[1]]])
        cv2.putText(img, "A", (4, 20), 0, 0.7, (0, 255, 255), 2); cv2.putText(img, "B", (4, rows["live"].shape[0] + 30), 0, 0.7, (0, 255, 255), 2)
        fn = f"pair_{k:03d}.jpg"; cv2.imwrite(f"{jdir}/{fn}", img)
        key.append({"file": fn, "clip": clip, "start": int(s), "A": order[0] if order[0] != "live" else base, "B": order[1] if order[1] != "live" else base}); k += 1
json.dump(key, open(f"{jdir}/key.json", "w"), indent=1)
print(jdir, k)
