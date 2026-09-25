import cProfile, pstats, sys, cv2, numpy as np
sys.path.insert(0, "..")
import dental as D
key, a, b = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
d = dict(np.load(f"cache/{key}.npz", allow_pickle=True))
cap = cv2.VideoCapture(str(d["path"]))
af = D.AnchorFill()
frames = []
n = 0
while n < b:
    ok, f = cap.read()
    if n >= a and d["ok"][n]:
        frames.append((f, d["pts"][n].astype(np.float64)))
    n += 1
for f, p in frames[:10]:
    af.process(f, p)
pr = cProfile.Profile()
pr.enable()
for f, p in frames:
    af.process(f, p)
pr.disable()
pstats.Stats(pr).sort_stats(sys.argv[4] if len(sys.argv) > 4 else "tottime").print_stats(18)
