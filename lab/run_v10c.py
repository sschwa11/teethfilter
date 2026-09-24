import os
import sys
import time

import cv2
import numpy as np
import torch

import common as C
import render_v10c as V
from run_v10 import mouth_box

OUT = os.path.join(C.HERE, "out")


class MouthSource:
    def __init__(self, kind):
        self.kind = kind
        self.df = C.dental()
        self.key = None
        self.last = None
        if kind == "bisenet":
            import parsers as P
            self.p = P.Parser("bisenet")
        elif kind == "teacher":
            pass
        else:
            from seg_model import SegNet
            w = int(kind.split("_w")[1]) if "_w" in kind else 16
            self.net = SegNet(w).cuda().eval()
            self.net.load_state_dict(torch.load(os.path.join(C.CACHE, kind + ".pt"), map_location="cuda"))

    def set_video(self, key):
        self.key = key
        if self.kind == "teacher":
            import build_atlas2 as B2
            _, self.tm, self.tok = B2.load_teach(key)

    @torch.no_grad()
    def __call__(self, frame, pts, n=None):
        if self.kind == "teacher":
            if n is not None and n < len(self.tok) and self.tok[n]:
                self.last = self.tm[n]
            return self.last if self.last is not None else np.zeros((C.CH, C.CW), np.float32)
        a0 = self.df.affine(self.df.pose(pts))
        if self.kind == "bisenet":
            p, m, _ = self.p.probs(frame, pts)
            return self.p.to_dental(p, m, a0)["mouth"]
        crop0 = cv2.warpAffine(frame, a0.astype(np.float32), (C.CW, C.CH), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
        x = torch.from_numpy(crop0[2:258, :, ::-1].copy()).cuda().permute(2, 0, 1)[None].float() / 255
        pr = torch.sigmoid(self.net(x))[0, 0].cpu().numpy()
        out = np.zeros((C.CH, C.CW), np.float32)
        out[2:258] = pr
        return out


def run(key, tag, mouth_src, make_renderer, compare=None, shots_every=37, tile_w=480):
    d = np.load(os.path.join(C.CACHE, key + ".npz"), allow_pickle=True)
    cap = cv2.VideoCapture(str(d["path"]))
    cmp_cap = cv2.VideoCapture(compare) if compare and os.path.exists(compare) else None
    r = make_renderer()
    if hasattr(mouth_src, "set_video"):
        mouth_src.set_video(key)
    os.makedirs(os.path.join(OUT, tag), exist_ok=True)
    writer = None
    full = None
    box = None
    shots, dbgs = {}, {}
    t_proc = []
    n = 0
    th = int(tile_w * 0.75 / 1.3)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cmp_frame = cmp_cap.read()[1] if cmp_cap is not None else None
        out = frame
        if d["ok"][n]:
            pts = d["pts"][n].astype(np.float64)
            mouth = mouth_src(frame, pts, n)
            t0 = time.perf_counter()
            out, dbg = r.process(frame, pts, mouth, debug=(n % shots_every == 0))
            t_proc.append(time.perf_counter() - t0)
            if dbg is not None:
                dbgs[n] = dbg
            b = mouth_box(pts, frame.shape)
            box = b if box is None else tuple(int(0.8 * p + 0.2 * q) for p, q in zip(box, b))
        if full is None:
            full = cv2.VideoWriter(os.path.join(OUT, tag, key + "_full.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (frame.shape[1], frame.shape[0]))
        full.write(out)
        if box is not None:
            x0, y0, w, h = box
            tiles = [cv2.resize(frame[y0:y0 + h, x0:x0 + w], (tile_w, th), interpolation=cv2.INTER_CUBIC)]
            if cmp_frame is not None:
                tiles.append(cv2.resize(cmp_frame[y0:y0 + h, x0:x0 + w], (tile_w, th), interpolation=cv2.INTER_CUBIC))
            tiles.append(cv2.resize(out[y0:y0 + h, x0:x0 + w], (tile_w, th), interpolation=cv2.INTER_CUBIC))
            sbs = np.hstack(tiles)
            cv2.putText(sbs, f"{n}", (4, 16), 0, 0.5, (0, 255, 0), 1)
            if writer is None:
                writer = cv2.VideoWriter(os.path.join(OUT, tag, key + ".mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (sbs.shape[1], sbs.shape[0]))
            writer.write(sbs)
            if n % shots_every == 0:
                shots[n] = sbs
        n += 1
    if writer is not None:
        writer.release()
    if full is not None:
        full.release()
    ks = sorted(shots)
    for part in range(0, len(ks), 8):
        cv2.imwrite(os.path.join(OUT, tag, f"{key}_shots{part // 8}.jpg"), np.vstack([shots[k] for k in ks[part:part + 8]]))
    rows = []
    for k in sorted(dbgs):
        g = dbgs[k]
        crop = g["crop"].copy()
        cs, _ = cv2.findContours((g["mouth"] > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(crop, cs, -1, (0, 255, 0), 1)
        al = cv2.cvtColor((g["alpha"] * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        t = np.hstack([crop[80:180, 40:280], g["comp"][80:180, 40:280], al[80:180, 40:280]])
        t = cv2.resize(t, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        cv2.putText(t, f"{k} conf {g['score']:.2f} d {np.round(g['delta'], 1)}", (4, 16), 0, 0.5, (0, 255, 255), 1)
        rows.append(t)
    for part in range(0, len(rows), 7):
        cv2.imwrite(os.path.join(OUT, tag, f"{key}_dbg{part // 7}.jpg"), np.vstack(rows[part:part + 7]))
    summ = r.summary() if hasattr(r, "summary") else {}
    print(f"{key}: {n} frames, render {1000 * np.mean(t_proc):.1f} ms/frame  " + "  ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in summ.items()), flush=True)
    return summ


if __name__ == "__main__":
    tag, src = sys.argv[1], sys.argv[2]
    keys = sys.argv[3:] or ["2026-09-23_05-46-00_raw"]
    model = V.Model()
    ms = MouthSource(src)
    for k in keys:
        run(k, tag, ms, lambda: V.Renderer(model), compare=os.path.join(OUT, "v9", k.replace("_raw", "") + ".mp4"))
