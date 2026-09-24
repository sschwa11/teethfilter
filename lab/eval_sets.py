import numpy as np, cv2, common as C, rigid as R
ref = np.load("cache/ref_shape.npy")
d0, r0 = C.load("2026-09-22_23-02-18")
df0 = C.dental()
tpl_img = C.warp_region(r0[148], df0.affine(df0.pose(d0["pts"][148])))
tpl = cv2.cvtColor(tpl_img, cv2.COLOR_BGR2GRAY).astype(np.float32)[118:142, 160:200]
TOP70 = np.load("cache/rigid_idx.npy")
NOSE_TIP = {1, 2, 3, 4, 5, 19, 44, 45, 51, 274, 275, 281, 248, 237, 238, 239, 241, 309, 457, 458, 459, 461, 354, 125, 141, 370, 94, 97, 326, 98, 327, 60, 290, 79, 20, 242, 250, 462}
BRIDGE = [i for i in TOP70 if i not in NOSE_TIP]
UPPER = [10, 151, 9, 8, 168, 6, 197, 133, 362, 33, 263, 127, 356, 234, 454, 21, 251, 109, 338, 67, 297, 103, 332, 54, 284, 107, 336, 55, 285, 193, 417, 122, 351, 245, 465, 188, 412, 114, 343, 196, 419, 174, 399]
SETS = {"top70": TOP70, "bridge": np.array(BRIDGE), "upper": np.array(UPPER), "bridge+upper": np.array(sorted(set(BRIDGE) | set(UPPER)))}
KEYS = [k for k in C.keys() if k not in ("2026-09-22_18-36-31", "2026-09-22_18-54-22")]
data = {k: C.load(k) for k in KEYS}


def evaluate(idx, use_mat=False):
    stds, jit = [], []
    for k in KEYS:
        d, regs = data[k]
        df = R.DentalFrame(ref, idx)
        res = []
        for n, reg in enumerate(regs):
            if reg is None:
                res.append((np.nan, np.nan, 0)); continue
            crop = C.warp_region(reg, df.affine(df.pose(d["pts"][n])))
            g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)[118 - 30:142 + 30, 160 - 30:200 + 30]
            r = cv2.matchTemplate(g, tpl, cv2.TM_CCOEFF_NORMED)
            _, mv, _, ml = cv2.minMaxLoc(r)
            res.append((ml[0] - 30, ml[1] - 30, mv))
        res = np.array(res)
        g = res[:, 2] > 0.85
        if g.sum() < 20:
            continue
        p = res[g, :2]
        med = np.median(p, 0)
        ok = np.all(np.abs(p - med) < 12, 1)
        p = p[ok]
        stds.append(np.std(p, 0))
        ii = np.flatnonzero(g)[ok]
        cons = ii[1:][np.diff(ii) == 1]
        pos = {i: q for i, q in zip(ii, p)}
        dd = np.array([pos[i] - pos[i - 1] for i in cons if i - 1 in pos])
        if len(dd):
            jit.append(np.mean(np.abs(dd), 0))
        print(f"   {k}: n={len(p)} med={med} std={np.std(p, 0).round(2)}")
    return np.mean(stds, 0), np.mean(jit, 0)


for name, idx in SETS.items():
    s, j = evaluate(idx)
    print(f"{name} ({len(idx)} pts): mean std {s.round(2)}  mean |frame delta| {j.round(2)}")
