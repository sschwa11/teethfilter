import argparse
import glob
import os
import sys
import time

import cv2
import numpy as np

import teeth_filter as T
import ai_model as AM

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "ai_data")
PREVIEW_DIR = os.path.join(HERE, "ai_preview")
MODEL_OUT = os.path.join(HERE, "teeth_ai.onnx")
Y0, Y1, X0, X1 = T.AI_CROP
H, W = Y1 - Y0, X1 - X0


def column_fill(mask):
    has = mask.any(0)
    if not has.any():
        return mask
    top = np.where(has, mask.argmax(0), mask.shape[0])
    bot = np.where(has, mask.shape[0] - 1 - mask[::-1].argmax(0), -1)
    rows = np.arange(mask.shape[0])[:, None]
    return (rows >= top[None, :]) & (rows <= bot[None, :])


def analyze(frame, pts):
    m = T.mouth_frame(pts)
    cp = T.to_canon(m, pts)
    yu, yl = T.lip_curves(cp)
    cx = slice(int(T.CW / 2 - T.MOUTH_PX * 0.2), int(T.CW / 2 + T.MOUTH_PX * 0.2))
    open_ratio = float(np.median((yl - yu)[cx])) / T.MOUTH_PX
    if open_ratio < 0.07:
        return None
    crop = cv2.warpAffine(frame, m.astype(np.float32), (T.CW, T.CH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    inside = cv2.erode(T.mouth_mask(cp), np.ones((3, 3), np.uint8)) > 127
    inside_up = T.upward(inside, int(0.05 * T.MOUTH_PX))
    t = T.otsu_split(lab[..., 0][inside & (np.abs(T.GU) < 0.85)])
    if t is None:
        return None
    bright = T.lip_free_bright(lab, inside_up, t, yu, yl)
    open_ = np.maximum(yl - yu, 1)
    labels = np.zeros((T.CH, T.CW), np.uint8)
    nid = 0
    for upper in (True, False):
        if upper:
            region = inside_up & (T.GY <= np.maximum(yu + 0.65 * open_, yu + 0.12 * T.MOUTH_PX)[None, :])
        else:
            region = inside_up & (T.GY >= np.minimum(yl - 0.65 * open_, yl - 0.12 * T.MOUTH_PX)[None, :])
        b = cv2.morphologyEx((bright & region).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, lab_ids, stats, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
        for i in range(1, n):
            x, y, w, h, area = [int(v) for v in stats[i]]
            if w < 0.06 * T.MOUTH_PX or h < 0.03 * T.MOUTH_PX or area < 0.002 * T.MOUTH_PX ** 2:
                continue
            cxi = min(max(x + w // 2, 0), T.CW - 1)
            if (inside & (lab_ids == i)).sum() < 0.5 * area:
                continue
            if upper and (y > yu[cxi] + 0.4 * open_[cxi] or y + h < yu[cxi] + 3):
                continue
            if not upper and y + h < yl[cxi] - 0.4 * open_[cxi]:
                continue
            nid += 1
            labels[lab_ids == i] = nid
            if nid >= 250:
                break
    if nid == 0:
        return None
    gaps = T.find_edge_mirror(bright, inside_up, yu, yl, type("C", (), {"min_gap": 0.025, "mirror_extent": 0.55})())
    gap = np.zeros((T.CH, T.CW), bool)
    hint = crop.copy()
    for g in gaps:
        rp, mk = T.fill_edge(g, crop)
        sel = g["mask"]
        hint[sel] = np.clip(rp[sel], 0, 255).astype(np.uint8)
        gap |= sel
    s = (slice(Y0, Y1), slice(X0, X1))
    return crop[s].copy(), labels[s].copy(), inside_up[s].astype(np.uint8), gap[s].astype(np.uint8), hint[s].copy()


def build_dataset(sources, step):
    mp_mod = None
    crops, labels, insides, gaps, hints = [], [], [], [], []
    for path in sources:
        mp_mod, lm = T.make_landmarker(True)
        cap = cv2.VideoCapture(path)
        n = kept = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            pts = T.detect(mp_mod, lm, frame, n * 33 + 1)
            if pts is not None and n % step == 0:
                r = analyze(frame, pts)
                if r is not None:
                    crops.append(r[0])
                    labels.append(r[1])
                    insides.append(r[2])
                    gaps.append(r[3])
                    hints.append(r[4])
                    kept += 1
            n += 1
        cap.release()
        lm.close()
        print(f"  {os.path.basename(path)}: {n} frames, {kept} usable")
    return np.array(crops), np.array(labels), np.array(insides), np.array(gaps), np.array(hints)


def make_mask(lab, inside, rng):
    ids = np.unique(lab)
    ids = ids[ids > 0]
    m = np.zeros(lab.shape, bool)
    axes = []
    if ids.size and rng.random() < 0.95:
        for _ in range(1 if rng.random() < 0.85 else 2):
            comp = lab == rng.choice(ids)
            cols = np.flatnonzero(comp.any(0))
            x0, x1 = int(cols.min()), int(cols.max())
            tw = int(rng.integers(16, 40))
            lo = x0 - tw // 3
            hi = max(lo + 1, x1 - tw + tw // 3)
            start = int(rng.integers(lo, hi + 1))
            end = min(lab.shape[1], start + tw)
            start = max(0, start)
            if end - start < 6:
                continue
            win = np.zeros(lab.shape, bool)
            win[:, start:end] = True
            m |= column_fill(comp) & win
            left_room = start - x0
            right_room = x1 - end
            if left_room >= right_room:
                axes.append((start - 0.5, start, end))
            else:
                axes.append((end - 0.5, start, end))
    if m.sum() < 40:
        cy, cx = int(rng.integers(20, H - 20)), int(rng.integers(40, W - 40))
        e = np.zeros(lab.shape, np.uint8)
        cv2.ellipse(e, (cx, cy), (int(rng.integers(8, 20)), int(rng.integers(6, 14))), 0, 0, 360, 1, -1)
        m = e > 0
        axes = []
    k = int(rng.integers(1, 3))
    m = cv2.dilate(m.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))) > 0
    return m & (cv2.dilate(inside, np.ones((7, 7), np.uint8)) > 0), axes


def make_hint(img, m, axes, rng):
    hint = img.copy()
    gx = np.arange(W, dtype=np.float32)
    if axes and rng.random() < 0.7:
        mapx = np.tile(gx[None, :], (H, 1))
        for ax, s0, s1 in axes:
            cols = slice(max(0, s0 - 3), min(W, s1 + 3))
            mapx[:, cols] = 2 * ax - gx[cols][None, :]
        mapy = np.tile(np.arange(H, dtype=np.float32)[:, None], (1, W)) + float(rng.integers(-2, 3))
        hint = cv2.remap(img, mapx.astype(np.float32), mapy.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    else:
        M = np.float32([[1, 0, rng.uniform(-3, 3)], [0, 1, rng.uniform(-2, 2)]])
        hint = cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_REPLICATE)
    hint = hint * rng.uniform(0.82, 1.18, 3).astype(np.float32) + rng.uniform(-0.05, 0.05)
    if rng.random() < 0.5:
        hint = cv2.GaussianBlur(hint, (0, 0), rng.uniform(0.3, 1.5))
    return np.clip(hint, 0, 1).astype(np.float32)


def augment(img, lab, ins, rng):
    if rng.random() < 0.5:
        img, lab, ins = img[:, ::-1], lab[:, ::-1], ins[:, ::-1]
    dx, dy = int(rng.integers(-8, 9)), int(rng.integers(-5, 6))
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    img = cv2.warpAffine(np.ascontiguousarray(img), M, (W, H), borderMode=cv2.BORDER_REPLICATE)
    lab = cv2.warpAffine(np.ascontiguousarray(lab), M, (W, H), flags=cv2.INTER_NEAREST)
    ins = cv2.warpAffine(np.ascontiguousarray(ins), M, (W, H), flags=cv2.INTER_NEAREST)
    f = img.astype(np.float32) / 255
    f = f * rng.uniform(0.8, 1.2, 3).astype(np.float32) + rng.uniform(-0.06, 0.06)
    f = np.clip(f, 0, 1) ** rng.uniform(0.8, 1.25)
    if rng.random() < 0.3:
        f = cv2.GaussianBlur(f, (0, 0), rng.uniform(0.4, 1.2))
    if rng.random() < 0.3:
        f = f + rng.normal(0, 0.01, f.shape).astype(np.float32)
    return np.clip(f, 0, 1).astype(np.float32), lab, ins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="*", default=None)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--iters", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--no-perceptual", dest="perceptual", action="store_false")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(PREVIEW_DIR, exist_ok=True)
    cache = os.path.join(DATA_DIR, "dataset_v2.npz")
    if args.rebuild or not os.path.exists(cache):
        sources = args.videos or sorted(glob.glob(os.path.join(HERE, "recordings", "*_raw.mp4")))
        if not sources:
            print("No videos found. Record some clips with [r] first.")
            return
        print(f"Building dataset from {len(sources)} videos...")
        T.ensure_model()
        crops, labels, insides, gaps, ghints = build_dataset(sources, args.step)
        if len(crops) < 50:
            print(f"Only {len(crops)} usable frames; record more talking/smiling clips.")
            return
        np.savez_compressed(cache, crops=crops, labels=labels, insides=insides, gaps=gaps, hints=ghints)
    d = np.load(cache)
    crops, labels, insides, gaps, ghints = d["crops"], d["labels"], d["insides"], d["gaps"], d["hints"]
    print(f"Dataset: {len(crops)} frames, {int((gaps.reshape(len(gaps), -1).sum(1) > 0).sum())} with real gaps (used for previews)")

    dev = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print("Training on", dev, torch.cuda.get_device_name(0) if dev.type == "cuda" else "")
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(crops))
    nval = max(8, len(crops) // 20)
    val_idx, tr_idx = idx[:nval], idx[nval:]

    net = AM.TeethNet(args.width).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.5, 0.999))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.iters, eta_min=args.lr * 0.05)

    vgg = None
    if args.perceptual:
        try:
            import torchvision
            vgg = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1).features[:16].to(dev).eval()
            for p in vgg.parameters():
                p.requires_grad_(False)
        except Exception as e:
            print("Perceptual loss disabled:", e)
            vgg = None
    vmean = torch.tensor([0.406, 0.456, 0.485], device=dev).view(1, 3, 1, 1)
    vstd = torch.tensor([0.225, 0.224, 0.229], device=dev).view(1, 3, 1, 1)

    def batch(ids, train=True):
        xs, ms, hs = [], [], []
        for i in ids:
            img, lab, ins = crops[i], labels[i], insides[i]
            if train:
                img, lab, ins = augment(img, lab, ins, rng)
            else:
                img = img.astype(np.float32) / 255
            m, axes = make_mask(lab, ins, rng)
            xs.append(img)
            ms.append(m.astype(np.float32))
            hs.append(make_hint(img, m, axes, rng))
        y = torch.from_numpy(np.stack(xs)).permute(0, 3, 1, 2).to(dev)
        m = torch.from_numpy(np.stack(ms))[:, None].to(dev)
        h = torch.from_numpy(np.stack(hs)).permute(0, 3, 1, 2).to(dev)
        return y, m, h

    def sobel(t):
        gx = t[..., :, 1:] - t[..., :, :-1]
        gy = t[..., 1:, :] - t[..., :-1, :]
        return gx, gy

    t0 = time.time()
    for it in range(1, args.iters + 1):
        net.train()
        y, m, h = batch(rng.choice(tr_idx, args.batch))
        pred = net(AM.ai_input(y, m, h))
        comp = pred * m + y * (1 - m)
        area = m.sum() * 3 + 1
        l_hole = (torch.abs(pred - y) * m).sum() / area
        l_all = torch.abs(pred - y).mean()
        gpx, gpy = sobel(comp)
        gyx, gyy = sobel(y)
        l_grad = (torch.abs(gpx - gyx) * m[..., :, 1:]).sum() / area + (torch.abs(gpy - gyy) * m[..., 1:, :]).sum() / area
        loss = 6 * l_hole + 0.5 * l_all + 2 * l_grad
        if vgg is not None:
            fa = vgg((comp.flip(1) - vmean) / vstd)
            fb = vgg((y.flip(1) - vmean) / vstd)
            loss = loss + 0.08 * torch.abs(fa - fb).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if it % 200 == 0 or it == 1:
            print(f"iter {it}/{args.iters}  loss {loss.item():.4f}  hole {l_hole.item():.4f}  {time.time() - t0:.0f}s", flush=True)
        if it % 2000 == 0 or it == args.iters:
            preview(net, crops, labels, insides, gaps, ghints, val_idx, dev, it, rng)
    net.eval()
    net.cpu()
    dummy = torch.zeros(1, 7, H, W)
    try:
        torch.onnx.export(net, dummy, MODEL_OUT, input_names=["x"], output_names=["y"], opset_version=17, dynamo=False)
    except TypeError:
        torch.onnx.export(net, dummy, MODEL_OUT, input_names=["x"], output_names=["y"], opset_version=17)
    torch.save(net.state_dict(), os.path.join(DATA_DIR, "teeth_ai.pt"))
    print("Saved", MODEL_OUT, "- restart run.bat to use it (press [m] to compare with the mirror fill).")


def preview(net, crops, labels, insides, gaps, ghints, val_idx, dev, it, rng):
    import torch
    net.eval()
    rows = []

    def row(img, m, hint):
        y = torch.from_numpy(img).permute(2, 0, 1)[None].to(dev)
        mt = torch.from_numpy(m.astype(np.float32))[None, None].to(dev)
        ht = torch.from_numpy(hint).permute(2, 0, 1)[None].to(dev)
        p = net(AM.ai_input(y, mt, ht))
        c = p * mt + y * (1 - mt)
        hv = ht * mt + y * (1 - mt)
        return np.hstack([(t[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8) for t in (y, hv, c)])

    with torch.no_grad():
        for i in val_idx[:5]:
            img = crops[i].astype(np.float32) / 255
            m, axes = make_mask(labels[i], insides[i], rng)
            rows.append(row(img, m, make_hint(img, m, axes, rng)))
        gi = np.flatnonzero(gaps.reshape(len(gaps), -1).sum(1) > 0)
        for i in (gi[np.linspace(0, len(gi) - 1, min(7, len(gi))).astype(int)] if gi.size else []):
            img = crops[i].astype(np.float32) / 255
            m = cv2.dilate(gaps[i], cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0
            rows.append(row(img, m, ghints[i].astype(np.float32) / 255))
    img = cv2.resize(np.vstack(rows), None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    path = os.path.join(PREVIEW_DIR, f"preview_{it:05d}.png")
    cv2.imwrite(path, img)
    print("  preview:", path)


if __name__ == "__main__":
    main()
