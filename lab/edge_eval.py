import sys, numpy as np, cv2, build_atlas2 as B, edge_track as E
keys = sys.argv[1:] or ["2026-09-23_05-47-41_raw", "2026-09-23_05-46-00_raw", "2026-09-22_23-02-18", "2026-09-22_23-36-22_raw", "2026-09-23_05-46-53_raw", "2026-09-23_05-47-17_raw"]
for key in keys:
    crops, mouth, ok = B.load_teach(key)
    t = E.EdgeTracker()
    xs, ys, idx, confs = [], [], [], []
    for p in range(2):
        for i in np.flatnonzero(ok):
            m = t.measure(crops[i], mouth[i])
            t.update(m)
            if p == 1 and m is not None and m["conf"] >= 0.12:
                xs.append(m["x"]); ys.append(m["y"] if m["y"] is not None else np.nan); idx.append(i); confs.append(m["conf"])
    y0, y1, x0, x1 = B.REF_BOX
    vis = np.array([mouth[i][100:140, 150:200].mean() if ok[i] else 0 for i in range(len(ok))])
    nvis = int((ok & (vis > 0.25)).sum())
    xs, ys, idx = np.array(xs), np.array(ys), np.array(idx)
    cons = np.flatnonzero(np.diff(idx) == 2)
    jx = np.median(np.abs(np.diff(xs)[cons])) if cons.size else np.nan
    print(f"{key}: meas {len(xs)} (visible {nvis}) off {t.off.round(1)} x med {np.median(xs):.1f} sd {np.std(xs):.2f} step|dx| med {jx:.2f}  y med {np.nanmedian(ys):.1f} sd {np.nanstd(ys):.2f} yvalid {np.isfinite(ys).sum()}")
    sel = idx[np.linspace(0, len(idx) - 1, 6).astype(int)] if len(idx) else []
    tiles = []
    for i in sel:
        c = cv2.resize(crops[i], None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        k = list(idx).index(i)
        cv2.line(c, (int(2 * xs[k]), 180), (int(2 * xs[k]), 300), (0, 255, 0), 1)
        if np.isfinite(ys[k]): cv2.line(c, (int(2 * xs[k]), int(2 * ys[k])), (int(2 * xs[k]) + 60, int(2 * ys[k])), (0, 0, 255), 1)
        tiles.append(c[170:330, 180:500])
    if tiles:
        cv2.imwrite(f"out/edge_{key[:19]}.jpg", np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:6])]))
