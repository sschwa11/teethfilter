import sys

import cv2
import numpy as np

import common as C
import render_v10 as V

CW, CH = C.CW, C.CH


def main(x_mid=157.5, span=88, tpl_box=(112, 144, 152, 236), key="2026-09-22_23-02-18", frame=148):
    z = np.load("cache/atlas_v1.npz")
    prob, mean, tooth = z["prob"], z["mean"], z["tooth"]
    df = C.dental()
    d, regs = C.load(key)
    crop = C.warp_region(regs[frame], df.affine(df.pose(d["pts"][frame])))
    g = V.highpass(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32))
    y0, y1, x0, x1 = tpl_box
    tpl = g[y0:y1, x0:x1]

    xs = np.arange(CW)
    bottom = np.zeros(CW, np.float32)
    top = np.zeros(CW, np.float32)
    for x in range(int(np.ceil(x_mid)), min(CW, int(x_mid + span) + 1)):
        col = prob[100:152, x]
        pk = 12 + int(np.argmax(col[12:45]))
        if col[pk] < 0.45:
            continue
        run_end = pk
        while run_end + 1 < len(col) and col[run_end + 1] > 0.45:
            run_end += 1
        run_start = pk
        while run_start > 0 and col[run_start - 1] > 0.45:
            run_start -= 1
        bottom[x] = 100 + run_end + 0.5
        top[x] = 100 + run_start
    donor_cols = np.flatnonzero(bottom > 0)
    good = bottom[donor_cols]
    k = 5
    padded = np.pad(good, k // 2, mode="edge")
    bottom[donor_cols] = np.median(np.lib.stride_tricks.sliding_window_view(padded, k), 1)
    for x in range(0, int(np.ceil(x_mid))):
        src = int(round(2 * x_mid - x))
        if 0 <= src < CW and bottom[src] > 0 and x_mid - x <= span:
            bottom[x] = bottom[src]
            top[x] = top[src]

    slot = np.zeros((CH, CW), bool)
    for x in range(CW):
        if x < x_mid and bottom[x] > 0:
            slot[int(max(bottom[x] - 45, 0)):int(bottom[x]) + 1, x] = True

    atlas = np.zeros((CH, CW, 3), np.float32)
    ok = np.zeros((CH, CW), bool)
    src_ok = (tooth > 20) & (prob > 0.3)
    gy, gx = np.mgrid[0:CH, 0:CW]
    sx = np.clip(np.round(2 * x_mid - gx).astype(int), 0, CW - 1)
    atlas[slot] = mean[gy[slot], sx[slot]]
    ok[slot] = src_ok[gy[slot], sx[slot]]
    for x in np.flatnonzero(slot.any(0)):
        rows = np.flatnonzero(ok[:, x] & slot[:, x])
        if rows.size == 0:
            continue
        r0 = rows[0]
        atlas[:r0, x] = atlas[r0, x]
        ok[:r0, x] = True

    np.savez_compressed("cache/model_v10.npz", tpl=tpl, tpl_box=np.array(tpl_box), x_mid=x_mid, bottom=bottom, top=top, slot=slot,
                        atlas=atlas, atlas_ok=ok, ref_box=np.array([118, 136, 162, 192]))
    print("bottom donor:", np.round(bottom[int(x_mid):int(x_mid) + span:4], 1))
    print("bottom slot:", np.round(bottom[int(x_mid) - span:int(x_mid):4], 1))
    vis = cv2.cvtColor(np.clip(np.dstack([atlas[..., 0] * 200, atlas[..., 1] + 132, atlas[..., 2] + 140]), 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    vis[~slot] = crop[~slot]
    big = cv2.resize(vis[80:170, 40:290], None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite("cache/model_v10.jpg", big)


if __name__ == "__main__":
    main()
