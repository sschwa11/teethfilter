import sys

import cv2
import numpy as np

import atlas as A
import common as C

INC_BOX = (118, 142, 160, 200)


def global_template(df):
    d, regs = C.load("2026-09-22_23-02-18")
    crop = C.warp_region(regs[148], df.affine(df.pose(d["pts"][148])))
    y0, y1, x0, x1 = INC_BOX
    return cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)[y0:y1, x0:x1]


def track_video(key, df, tracker, d=None, regs=None):
    if d is None:
        d, regs = C.load(key)
    n = len(regs)
    res = np.full((n, 3), np.nan)
    for i, reg in enumerate(regs):
        if reg is None:
            continue
        crop = C.warp_region(reg, df.affine(df.pose(d["pts"][i])))
        res[i] = tracker.match(crop)
    good = res[:, 2] > 0.85
    if good.sum() >= 10:
        med = np.nanmedian(res[good, :2], 0)
        good &= np.all(np.abs(res[:, :2] - med) < 12, 1)
    delta = np.zeros((n, 2))
    idx = np.flatnonzero(good)
    if idx.size:
        for j in range(2):
            delta[:, j] = np.interp(np.arange(n), idx, res[idx, j])
    return res, good, delta


def accumulate(key, df, tracker, acc, min_open=18):
    d, regs = C.load(key)
    res, good, delta = track_video(key, df, tracker, d, regs)
    y0, y1, x0, x1 = INC_BOX
    used = 0
    for i, reg in enumerate(regs):
        if reg is None or not good[i]:
            continue
        pose = df.pose(d["pts"][i])
        a = A.shifted(df.affine(pose), *delta[i])
        crop = C.warp_region(reg, a)
        cp = df.to_crop(pose, d["pts"][i]) - delta[i]
        if np.linalg.norm(cp[13] - cp[14]) < min_open:
            continue
        bright, inside_up, lab = A.teeth_mask(crop, cp)
        if bright is None:
            continue
        inc = bright[y0:y1, x0:x1]
        if inc.sum() < 80:
            continue
        ref = np.array([np.median(lab[y0:y1, x0:x1, c][inc]) for c in range(3)])
        norm = np.dstack([lab[..., 0] / max(ref[0], 1), lab[..., 1] - ref[1], lab[..., 2] - ref[2]])
        w = bright.astype(np.float32)
        acc["sum"] += norm * w[..., None]
        acc["sq"] += norm ** 2 * w[..., None]
        acc["tooth"] += w
        acc["vis"] += inside_up.astype(np.float32)
        acc["refs"].append(ref)
        used += 1
    return used, res, good


def new_acc():
    return {"sum": np.zeros((C.CH, C.CW, 3)), "sq": np.zeros((C.CH, C.CW, 3)), "tooth": np.zeros((C.CH, C.CW)), "vis": np.zeros((C.CH, C.CW)), "refs": []}


def finish(acc):
    t = np.maximum(acc["tooth"], 1)[..., None]
    mean = acc["sum"] / t
    prob = acc["tooth"] / np.maximum(acc["vis"], 1)
    return mean, prob, acc["tooth"], acc["vis"]


def render_lab(mean, ref):
    lab = np.dstack([mean[..., 0] * ref[0], mean[..., 1] + ref[1], mean[..., 2] + ref[2]])
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


if __name__ == "__main__":
    df = C.dental()
    tracker = A.Tracker(global_template(df), INC_BOX)
    keys = sys.argv[1:] or C.keys()
    tiles = []
    allacc = new_acc()
    for k in keys:
        acc = new_acc()
        used, res, good = accumulate(k, df, tracker, acc)
        mean, prob, tooth, vis = finish(acc)
        for f in ("sum", "sq", "tooth", "vis"):
            allacc[f] += acc[f]
        allacc["refs"] += acc["refs"]
        print(f"{k}: tracked {good.sum()}/{len(good)}, atlas frames {used}", flush=True)
        if used < 5:
            continue
        img = render_lab(mean, np.array([200.0, 132.0, 140.0]))
        img[tooth < 3] = 0
        vis_img = cv2.cvtColor((np.clip(prob, 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        t = np.hstack([img[60:200, 40:280], vis_img[60:200, 40:280]])
        cv2.putText(t, k[:19], (4, 14), 0, 0.4, (0, 255, 0), 1)
        tiles.append(t)
    mean, prob, tooth, vis = finish(allacc)
    np.savez_compressed("cache/atlas_v1.npz", mean=mean, prob=prob, tooth=tooth, vis=vis, refs=np.array(allacc["refs"]))
    img = render_lab(mean, np.array([200.0, 132.0, 140.0]))
    img[tooth < 3] = 0
    vis_img = cv2.cvtColor((np.clip(prob, 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    tiles.append(np.hstack([img[60:200, 40:280], vis_img[60:200, 40:280]]))
    cv2.imwrite("cache/atlas_v1.jpg", np.vstack(tiles))
