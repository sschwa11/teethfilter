import sys, cv2, numpy as np, common as C, lips as Lp
df = C.dental()
tiles = []
for key, frames in (("2026-09-23_05-47-41_raw", [40, 120, 200, 300, 420, 470]), ("2026-09-22_23-36-22_raw", [30, 100, 200, 300]), ("2026-09-22_23-02-18", [148, 400, 600, 700, 820, 900])):
    d, regs = C.load(key)
    for n in frames:
        if regs[n] is None: continue
        pose = df.pose(d["pts"][n])
        crop = C.warp_region(regs[n], df.affine(pose))
        cp = df.to_crop(pose, d["pts"][n])
        yu, yl, lo, hi = Lp.refine(crop, cp)
        yu0, _, _ = Lp.curve(cp, Lp.UPPER_INNER, C.CW); yl0, _, _ = Lp.curve(cp, Lp.LOWER_INNER, C.CW)
        big = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        for x in range(lo, hi):
            for y, col in ((yu0[x], (0, 0, 255)), (yl0[x], (0, 0, 255)), (yu[x], (0, 255, 0)), (yl[x], (0, 255, 0))):
                cv2.circle(big, (2 * x, int(2 * y)), 0, col, -1)
        tiles.append(big[160:420, 80:560])
rows = [np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles) - 1, 2)]
cv2.imwrite("cache/lips_viz.jpg", np.vstack(rows))
