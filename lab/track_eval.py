import numpy as np, cv2, common as C, atlas as A, build_atlas as B

df = C.dental()
d0, r0 = C.load("2026-09-22_23-02-18")
key_crop = C.warp_region(r0[148], df.affine(df.pose(d0["pts"][148])))


def hp(img, s=3.0):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) if img.ndim == 3 else img
    return g - cv2.GaussianBlur(g, (0, 0), s)


def tracker_box(box, highpass, src=key_crop):
    y0, y1, x0, x1 = box
    tpl = (hp(src) if highpass else cv2.cvtColor(src, cv2.COLOR_BGR2GRAY).astype(np.float32))[y0:y1, x0:x1]
    return tpl


def run(key, box, highpass, thr, tpl=None, pad=24):
    d, regs = C.load(key)
    y0, y1, x0, x1 = box
    tpl = tracker_box(box, highpass) if tpl is None else tpl
    res = np.full((len(regs), 3), np.nan)
    crops = {}
    for i, reg in enumerate(regs):
        if reg is None:
            continue
        p = d["pts"][i]
        crop = C.warp_region(reg, df.affine(df.pose(p)))
        g = hp(crop) if highpass else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
        r = cv2.matchTemplate(g[y0 - pad:y1 + pad, x0 - pad:x1 + pad], tpl, cv2.TM_CCOEFF_NORMED)
        _, mv, _, ml = cv2.minMaxLoc(r)
        res[i] = (ml[0] - pad, ml[1] - pad, mv)
        crops[i] = crop
    opn = np.array([np.linalg.norm(p[13, :2] - p[14, :2]) / max(np.linalg.norm(p[78, :2] - p[308, :2]), 1) for p in d["pts"]])
    vis = opn > 0.12
    acc = (res[:, 2] > thr)
    if acc.sum() > 5:
        med = np.nanmedian(res[acc, :2], 0)
        acc &= np.all(np.abs(res[:, :2] - med) < 14, 1)
    idx = np.flatnonzero(acc)
    jit = []
    for j in idx:
        nb = [k for k in idx if 0 < abs(k - j) <= 2]
        if len(nb) >= 2:
            jit.append(np.abs(res[j, :2] - np.median(res[nb, :2], 0)))
    jit = np.mean(jit, 0) if jit else np.array([np.nan, np.nan])
    return acc, vis, res, jit, crops


if __name__ == "__main__":
    keys = [k for k in C.keys() if k not in ("2026-09-22_18-36-31", "2026-09-22_18-54-22", "2026-09-23_04-53-09")]
    configs = {"central-gray-0.85": ((118, 142, 160, 200), False, 0.85), "central-hp-0.6": ((116, 144, 156, 202), True, 0.6),
               "arch-hp-0.6": ((110, 144, 150, 240), True, 0.6), "arch-hp-0.5": ((110, 144, 150, 240), True, 0.5)}
    for name, (box, hpf, thr) in configs.items():
        tot_acc = tot_vis = 0
        js = []
        for k in keys:
            acc, vis, res, jit, _ = run(k, box, hpf, thr)
            tot_acc += (acc & vis).sum(); tot_vis += vis.sum()
            js.append(jit)
            print(f"  {name} {k}: accept {int((acc & vis).sum())}/{int(vis.sum())} jitter {np.round(jit, 2)}", flush=True)
        print(f"{name}: accept {tot_acc}/{tot_vis} = {tot_acc / tot_vis:.2f}  mean jitter {np.nanmean(js, 0).round(2)}", flush=True)
