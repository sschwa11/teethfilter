import glob, os, pickle, cv2, numpy as np

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")


def run(key):
    out = os.path.join(CACHE, key + "_reg.pkl")
    if os.path.exists(out):
        return
    d = np.load(os.path.join(CACHE, key + ".npz"), allow_pickle=True)
    cap = cv2.VideoCapture(str(d["path"]))
    regs = []
    n = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if d["ok"][n]:
            p = d["pts"][n]
            iod = np.linalg.norm(p[33, :2] - p[263, :2])
            c = p[[13, 14, 78, 308], :2].mean(0)
            x0 = int(max(c[0] - 0.9 * iod, 0)); x1 = int(min(c[0] + 0.9 * iod, f.shape[1]))
            y0 = int(max(c[1] - 0.75 * iod, 0)); y1 = int(min(c[1] + 0.65 * iod, f.shape[0]))
            regs.append((x0, y0, f[y0:y1, x0:x1].copy()))
        else:
            regs.append(None)
        n += 1
    pickle.dump(regs, open(out, "wb"), protocol=4)
    print(key, n, flush=True)


if __name__ == "__main__":
    for p in sorted(glob.glob(os.path.join(CACHE, "*.npz"))):
        k = os.path.splitext(os.path.basename(p))[0]
        if np.load(p)["ok"].sum() >= 60:
            run(k)
