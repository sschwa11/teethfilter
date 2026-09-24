import sys, time, cv2, numpy as np
sys.path.insert(0, "..")
import dental as D
key = sys.argv[1]
d = dict(np.load(f"cache/{key}.npz", allow_pickle=True))
cap = cv2.VideoCapture(str(d["path"]))
af = D.AnchorFill()
print("provider", af.seg.provider)
w = None; ts = []; n = 0
while True:
    ok, f = cap.read()
    if not ok: break
    out = f
    if d["ok"][n]:
        t = time.perf_counter()
        out, _ = af.process(f, d["pts"][n].astype(np.float64))
        ts.append(time.perf_counter() - t)
    if w is None:
        w = cv2.VideoWriter(f"out/dental_{key}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (f.shape[1], f.shape[0]))
    w.write(out); n += 1
w.release()
ts = np.array(ts[10:]) * 1000
print(f"{key}: {n} frames, process ms mean {ts.mean():.1f} p95 {np.percentile(ts, 95):.1f} max {ts.max():.1f}; offset {af.offset.round(1)}")
