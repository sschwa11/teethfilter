import sys, cv2, numpy as np
from run_v10 import mouth_box
key, tag, every, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
d = dict(np.load(f"cache/{key}.npz", allow_pickle=True))
base = key[:-4]
caps = [cv2.VideoCapture(f"../recordings/{base}_raw.mp4"), cv2.VideoCapture(f"../recordings/{base}_filtered.mp4"), cv2.VideoCapture(f"out/{tag}/{key}_full.mp4")]
rows = []; n = 0
while True:
    fr = [c.read()[1] for c in caps]
    if any(f is None for f in fr): break
    if n % every == 0 and d["ok"][n]:
        x0, y0, w, h = mouth_box(d["pts"][n], fr[0].shape, 1.0)
        th = int(320 * 0.75 / 1.3)
        t = np.hstack([cv2.resize(f[y0:y0+h, x0:x0+w], (320, th), interpolation=cv2.INTER_CUBIC) for f in fr])
        cv2.putText(t, f"{n}  raw | live | new", (4, 14), 0, 0.45, (0, 255, 0), 1)
        rows.append(t)
    n += 1
for part in range(0, len(rows), 10):
    cv2.imwrite(out.replace(".jpg", f"_{part // 10}.jpg"), np.vstack(rows[part:part + 10]))
