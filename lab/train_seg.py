import argparse
import glob
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import common as C
from seg_model import SegNet

VAL_KEYS = ["2026-09-23_05-47-17_raw", "2026-09-22_23-36-51_raw", "2026-09-27_00-24-05_raw", "2026-09-27_01-10-00_raw"]


def load(keys):
    xs, ys = [], []
    for k in keys:
        z = np.load(os.path.join(C.CACHE, k + "_teach.npz"))
        ok = z["ok"]
        xs.append(z["crops"][ok][:, 2:258])
        lab = z["labels"][ok][:, :, 0, 2:258].astype(np.float32)
        ys.append((0.6 * lab[:, 0] + 0.4 * lab[:, 1]).astype(np.uint8))
    return np.concatenate(xs), np.concatenate(ys)


def augment(x, y, gen):
    b = x.shape[0]
    dev = x.device
    ang = (torch.rand(b, device=dev, generator=gen) - 0.5) * 2 * math.radians(5)
    sc = 1 + (torch.rand(b, device=dev, generator=gen) - 0.5) * 0.16
    tx = (torch.rand(b, device=dev, generator=gen) - 0.5) * 2 * 10 / 160
    ty = (torch.rand(b, device=dev, generator=gen) - 0.5) * 2 * 12 / 128
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    theta = torch.stack([torch.stack([cos, -sin * 256 / 320, tx], 1), torch.stack([sin * 320 / 256, cos, ty], 1)], 1)
    flip = torch.rand(b, device=dev, generator=gen) < 0.5
    theta[flip, 0, :] *= torch.tensor([-1.0, -1.0, -1.0], device=dev)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)
    y = F.grid_sample(y, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    gain = 1 + (torch.rand(b, 3, 1, 1, device=dev, generator=gen) - 0.5) * 0.4
    bias = (torch.rand(b, 1, 1, 1, device=dev, generator=gen) - 0.5) * 0.2
    x = x * gain + bias
    gamma = torch.exp((torch.rand(b, 1, 1, 1, device=dev, generator=gen) - 0.5) * 0.6)
    x = x.clamp(1e-4, 1) ** gamma
    sat = 1 + (torch.rand(b, 1, 1, 1, device=dev, generator=gen) - 0.5) * 0.8
    g = x.mean(1, keepdim=True)
    x = g + (x - g) * sat
    blur = torch.rand(b, device=dev, generator=gen) < 0.4
    if blur.any():
        k = torch.tensor([1, 4, 6, 4, 1], device=dev, dtype=x.dtype)
        k = (k[:, None] * k[None, :]) / 256
        xb = F.conv2d(F.pad(x[blur], (2, 2, 2, 2), mode="replicate"), k.expand(3, 1, 5, 5), groups=3)
        x[blur] = xb
    x = x + torch.randn(x.shape, device=dev, generator=gen) * 0.015 * torch.rand(b, 1, 1, 1, device=dev, generator=gen)
    return x.clamp(0, 1), y


def to_t(x, y, dev):
    xt = torch.from_numpy(x).to(dev).permute(0, 3, 1, 2).float() / 255
    yt = torch.from_numpy(y).to(dev)[:, None].float() / 255
    return xt[:, [2, 1, 0]], yt


def evaluate(net, xv, yv, dev):
    net.eval()
    ious, maes, berr = [], [], []
    with torch.no_grad():
        for i in range(0, len(xv), 64):
            x, y = to_t(xv[i:i + 64], yv[i:i + 64], dev)
            p = torch.sigmoid(net(x))
            maes.append(torch.abs(p - y).mean().item())
            pb, yb = p > 0.5, y > 0.5
            inter = (pb & yb).sum((1, 2, 3)).float()
            uni = (pb | yb).sum((1, 2, 3)).float()
            sel = uni > 200
            if sel.any():
                ious.append((inter[sel] / uni[sel]).mean().item())
            cols = slice(100, 220)
            has = yb[:, 0, :, cols].any(1) & pb[:, 0, :, cols].any(1)
            ty = yb[:, 0, :, cols].float().argmax(1)
            tp = pb[:, 0, :, cols].float().argmax(1)
            if has.any():
                berr.append((ty - tp).abs()[has].float().mean().item())
    net.train()
    return float(np.mean(ious)), float(np.mean(maes)), float(np.mean(berr))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--out", default="seg_w16")
    args = ap.parse_args()
    keys = [os.path.basename(p)[:-10] for p in sorted(glob.glob(os.path.join(C.CACHE, "*_teach.npz")))]
    tr_keys = [k for k in keys if k not in VAL_KEYS]
    va_keys = [k for k in keys if k in VAL_KEYS]
    xt, yt = load(tr_keys)
    xv, yv = load(va_keys)
    print(f"train {len(xt)} val {len(xv)}", flush=True)
    dev = torch.device("cuda")
    net = SegNet(args.width).to(dev)
    print("params", sum(p.numel() for p in net.parameters()))
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.iters, pct_start=0.05)
    gen = torch.Generator(device=dev)
    gen.manual_seed(0)
    rng = np.random.default_rng(0)
    t0 = time.time()
    for it in range(1, args.iters + 1):
        idx = rng.integers(0, len(xt), args.batch)
        x, y = to_t(xt[idx], yt[idx], dev)
        x, y = augment(x, y, gen)
        logit = net(x)
        p = torch.sigmoid(logit)
        gy = torch.abs(F.max_pool2d(y, 5, 1, 2) + F.max_pool2d(-y, 5, 1, 2))
        w = 1 + 4 * gy
        loss = (F.binary_cross_entropy_with_logits(logit, y, reduction="none") * w).mean()
        inter = (p * y).sum((1, 2, 3))
        dice = 1 - (2 * inter + 1) / (p.sum((1, 2, 3)) + y.sum((1, 2, 3)) + 1)
        loss = loss + 0.3 * dice.mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if it % 1000 == 0 or it == args.iters:
            iou, mae, be = evaluate(net, xv, yv, dev)
            print(f"it {it} loss {loss.item():.4f} val IoU {iou:.3f} MAE {mae:.4f} top-edge err {be:.2f}px  {time.time() - t0:.0f}s", flush=True)
    torch.save(net.state_dict(), os.path.join(C.CACHE, args.out + ".pt"))


if __name__ == "__main__":
    main()
