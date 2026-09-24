import sys, cv2, numpy as np, rigid as R
key = sys.argv[1]
d = np.load(f"cache/{key}.npz", allow_pickle=True)
ref = np.load("cache/ref_shape.npy"); idx = np.load("cache/rigid_idx.npy")
df = R.DentalFrame(ref, idx)
cap = cv2.VideoCapture(str(d["path"]))
crops, lipc = [], []
n = 0
while True:
    ok, f = cap.read()
    if not ok: break
    if d["ok"][n]:
        pose = df.pose(d["pts"][n])
        crops.append(df.warp(f, pose))
        lipc.append(df.to_crop(pose, d["pts"][n][[13, 14, 78, 308]]))
    else:
        crops.append(None); lipc.append(None)
    n += 1
np.save(f"cache/{key}_lipc.npy", np.array([l if l is not None else np.zeros((4, 2)) for l in lipc]))
import pickle
pickle.dump(crops, open(f"cache/{key}_crops.pkl", "wb"))
op = np.array([np.linalg.norm(l[0] - l[1]) if l is not None else 0 for l in lipc])
print("frames", n, "open>25px", int((op > 25).sum()))
start = int(sys.argv[2]) if len(sys.argv) > 2 else int(np.argmax(op))
seq = [crops[i][60:200, 40:280] for i in range(start, min(start + 16, n)) if crops[i] is not None]
cv2.imwrite(f"cache/{key}_seq.jpg", np.vstack([np.hstack(seq[i:i + 4]) for i in range(0, len(seq) - 3, 4)]))
print("start", start)
