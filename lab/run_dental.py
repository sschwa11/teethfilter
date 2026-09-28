import os, sys, time, cv2, numpy as np
sys.path.insert(0, "..")
import dental as D
tag = sys.argv[1]
opts = [a for a in sys.argv[2:] if "=" in a]
keys = [a for a in sys.argv[2:] if "=" not in a]
os.makedirs(f"out/{tag}", exist_ok=True)
for key in keys:
    d = dict(np.load(f"cache/{key}.npz", allow_pickle=True))
    cap = cv2.VideoCapture(str(d["path"]))
    af = D.AnchorFill()
    if os.environ.get("LIVE"):
        import json
        af.reset(json.load(open("../settings.json"))["anchor_offset"])
        af.load_state()
    for o in opts:
        k, v = o.split("="); setattr(af, k, eval(v))
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
            w = cv2.VideoWriter(f"out/{tag}/{key}_full.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (f.shape[1], f.shape[0]))
        w.write(out); n += 1
    w.release()
    print(f"{key}: {n} frames {1000 * np.mean(ts):.1f} ms", flush=True)
