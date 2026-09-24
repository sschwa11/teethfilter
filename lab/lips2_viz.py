import cv2, numpy as np, common as C, lips2 as Lp
df = C.dental()
tiles = []
for key, frames in (("2026-09-23_05-47-41_raw", [40, 120, 200, 300, 420, 470]), ("2026-09-22_23-36-22_raw", [30, 100, 200, 300]), ("2026-09-22_23-02-18", [148, 400, 600, 700, 820, 900])):
    d, regs = C.load(key)
    for n in frames:
        if regs[n] is None: continue
        pose = df.pose(d["pts"][n])
        crop = C.warp_region(regs[n], df.affine(pose))
        cp = df.to_crop(pose, d["pts"][n])
        yu, yl, lo, hi, p = Lp.refine(crop, cp)
        yu0, _, _ = Lp.curve(cp, Lp.UPPER_INNER, C.CW); yl0, _, _ = Lp.curve(cp, Lp.LOWER_INNER, C.CW)
        big = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        pv = cv2.resize(cv2.cvtColor(((p if p is not None else np.zeros(crop.shape[:2])) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR), None, fx=2, fy=2)
        for x in range(lo, hi):
            for y, col in ((yu0[x], (0, 0, 255)), (yl0[x], (0, 0, 255)), (yu[x], (0, 255, 0)), (yl[x], (0, 255, 0))):
                cv2.circle(big, (2 * x, int(2 * y)), 0, col, -1)
        tiles.append(np.hstack([big[170:410, 80:560], pv[170:410, 80:560]]))
cv2.imwrite("cache/lips2_viz.jpg", np.vstack(tiles[:8]))
cv2.imwrite("cache/lips2_viz_b.jpg", np.vstack(tiles[8:]))
