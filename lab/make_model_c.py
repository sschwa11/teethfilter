import os

import cv2
import numpy as np

import common as C

CW, CH = C.CW, C.CH
X_E, B_E = 160.0, 137.0


def main(axis_gap=0.75, span=88, margin=2.5, crown=44):
    z = np.load(os.path.join(C.CACHE, "atlas_v3.npz"))
    prob, atlas, wsum = z["prob"], z["atlas"], z["wsum"]
    x_mid = X_E - axis_gap
    bottom = np.zeros(CW, np.float32)
    for xd in range(int(X_E), min(CW, int(X_E + span))):
        col = prob[96:152, xd]
        pk = int(np.argmax(col[10:45])) + 10
        if col[pk] < 0.3:
            continue
        thr = 0.5 * col[pk]
        e = pk
        while e + 1 < len(col) and col[e + 1] > thr:
            e += 1
        bottom[xd] = 96 + e + 0.5
    cols = np.flatnonzero(bottom > 0)
    span_cols = np.arange(int(X_E), int(X_E) + span)
    inside = span_cols[(span_cols >= cols.min()) & (span_cols <= cols.max())]
    bottom[inside] = np.interp(inside, cols, bottom[cols])
    cols = np.flatnonzero(bottom > 0)
    k = 5
    pad = np.pad(bottom[cols], k // 2, mode="edge")
    bottom[cols] = np.median(np.lib.stride_tricks.sliding_window_view(pad, k), 1)
    gy, gx = np.mgrid[0:CH, 0:CW].astype(np.float32)
    xd_f = 2 * x_mid - gx[0]
    b_slot = np.interp(xd_f, np.arange(CW), bottom, left=0, right=0)
    valid = (b_slot > 0) & (gx[0] < x_mid) & (x_mid - gx[0] <= span)
    slot = np.clip(b_slot[None, :] + margin - gy + 0.5, 0, 1) * (gy >= b_slot[None, :] - crown) * valid[None, :]
    slot[:, valid] *= np.clip((x_mid - gx[0][valid]) / 1.0 + 0.5, 0, 1)[None, :]
    atlas_slot = cv2.remap(atlas, xd_f[None, :].repeat(CH, 0).astype(np.float32), gy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    wmap = cv2.remap(wsum.astype(np.float32), xd_f[None, :].repeat(CH, 0).astype(np.float32), gy, cv2.INTER_LINEAR)
    for x in np.flatnonzero(valid):
        rows = np.flatnonzero(wmap[:, x] > 1)
        if rows.size:
            atlas_slot[:rows[0], x] = atlas_slot[rows[0], x]
    donor = (prob > 0.6) & (gx >= X_E + 2) & (gx < X_E + 60) & (gy >= 104) & (gy <= bottom[None, :] - 2)
    np.savez_compressed(os.path.join(C.CACHE, "model_v10c.npz"), x_mid=x_mid, edge_offset=axis_gap, bottom=bottom, slot=slot.astype(np.float32),
                        atlas_bgr=atlas.astype(np.float32), atlas_slot=atlas_slot.astype(np.float32), donor_mask=donor.astype(np.float32),
                        tpl=np.zeros((1, 1), np.float32), tpl_box=np.array([0, 1, 0, 1]), ref_box=np.array([122, 134, 164, 190]))
    print("bottom", bottom[int(X_E):int(X_E) + span:4].round(1))
    vis = np.clip(atlas_slot, 0, 255).astype(np.uint8)
    base = np.clip(atlas, 0, 255).astype(np.uint8)
    comp = (base * (1 - slot[..., None]) + vis * slot[..., None]).astype(np.uint8)
    cv2.imwrite(os.path.join(C.HERE, "out", "model_v10c.jpg"), cv2.resize(np.vstack([comp[80:170, 40:290], (np.dstack([slot] * 3)[80:170, 40:290] * 255).astype(np.uint8)]), None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST))


if __name__ == "__main__":
    main()
