import sys, cv2, numpy as np
sys.path.insert(0, "..")
import dental as D

CLIPS = ["2026-09-25_02-36-08_raw", "2026-09-25_01-18-25_raw", "2026-09-25_01-10-45_raw", "2026-09-25_00-43-21_raw", "2026-09-24_06-33-17_raw"]


def run(key, opts):
    d = dict(np.load(f"cache/{key}.npz", allow_pickle=True))
    cap = cv2.VideoCapture(str(d["path"]))
    af = D.AnchorFill(offset=[-0.14, -13.71])
    af.upper = False
    af.lower = False
    for k, v in opts.items():
        setattr(af, k, v)
    meas = []
    delta = []
    orig = af.et.measure
    def spy(L, mouth):
        m = orig(L, mouth)
        meas.append(m["x"] - af.et.x0 if (m and m["conf"] >= 0.12) else np.nan)
        return m
    af.et.measure = spy
    n = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if d["ok"][n]:
            k0 = len(meas)
            af.process(f, d["pts"][n].astype(np.float64))
            if len(meas) == k0:
                meas.append(np.nan)
            delta.append(af.delta[0] if af.delta is not None else np.nan)
        n += 1
    return np.array(meas), np.array(delta)


def score(meas, delta, w=15):
    truth = np.full_like(meas, np.nan)
    for i in range(len(meas)):
        seg = meas[max(0, i - w):i + w + 1]
        seg = seg[np.isfinite(seg)]
        if seg.size >= 5:
            truth[i] = np.median(seg)
    ok = np.isfinite(truth) & np.isfinite(delta)
    err = np.abs(delta - truth)[ok]
    jump = np.abs(np.diff(delta))
    jump = jump[np.isfinite(jump)]
    return err.mean(), (err > 3).mean(), jump.mean(), (jump > 2).mean()


if __name__ == "__main__":
    opts = {k: eval(v) for k, v in (a.split("=") for a in sys.argv[1:])}
    tot = []
    for c in CLIPS:
        s = score(*run(c, opts))
        tot.append(s)
        print(f"{c[:19]} err {s[0]:.2f} err>3 {s[1]:.3f} jump {s[2]:.2f} jump>2 {s[3]:.3f}")
    t = np.mean(tot, 0)
    print(f"MEAN err {t[0]:.2f} err>3 {t[1]:.3f} jump {t[2]:.2f} jump>2 {t[3]:.3f}")
