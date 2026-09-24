import os
import sys
import time

import cv2
import numpy as np

import common as C
import render_v10 as V

OUT = os.path.join(C.HERE, "out")


def mouth_box(pts, frame_shape, scale=1.0):
    c = pts[[13, 14, 78, 308], :2].mean(0)
    iod = np.linalg.norm(pts[33, :2] - pts[263, :2])
    w, h = 1.3 * iod * scale, 0.75 * iod * scale
    x0 = int(np.clip(c[0] - w / 2, 0, frame_shape[1] - w))
    y0 = int(np.clip(c[1] - h / 2, 0, frame_shape[0] - h))
    return x0, y0, int(w), int(h)


def run(key, tag, renderer_factory, frames=None, compare=None, write_video=True, tile_w=480):
    d = np.load(os.path.join(C.CACHE, key + ".npz"), allow_pickle=True)
    cap = cv2.VideoCapture(str(d["path"]))
    cmp_cap = cv2.VideoCapture(compare) if compare and os.path.exists(compare) else None
    r = renderer_factory()
    os.makedirs(os.path.join(OUT, tag), exist_ok=True)
    writer = None
    n = 0
    box = None
    shots = {}
    t_proc = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cmp_frame = cmp_cap.read()[1] if cmp_cap is not None else None
        out = frame
        if d["ok"][n]:
            pts = d["pts"][n].astype(np.float64)
            t0 = time.perf_counter()
            out, _ = r.process(frame, pts)
            t_proc.append(time.perf_counter() - t0)
            b = mouth_box(pts, frame.shape)
            box = b if box is None else tuple(int(0.8 * p + 0.2 * q) for p, q in zip(box, b))
        if box is not None and write_video:
            x0, y0, w, h = box
            th = int(tile_w * 0.75 / 1.3)
            tiles = [cv2.resize(frame[y0:y0 + h, x0:x0 + w], (tile_w, th), interpolation=cv2.INTER_CUBIC)]
            if cmp_frame is not None:
                tiles.append(cv2.resize(cmp_frame[y0:y0 + h, x0:x0 + w], (tile_w, th), interpolation=cv2.INTER_CUBIC))
            tiles.append(cv2.resize(out[y0:y0 + h, x0:x0 + w], (tile_w, th), interpolation=cv2.INTER_CUBIC))
            sbs = np.hstack(tiles)
            cv2.putText(sbs, f"{n}", (4, 16), 0, 0.5, (0, 255, 0), 1)
            if writer is None:
                writer = cv2.VideoWriter(os.path.join(OUT, tag, key + ".mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (sbs.shape[1], sbs.shape[0]))
            writer.write(sbs)
            if frames is not None and n in frames:
                shots[n] = sbs
        n += 1
    if writer is not None:
        writer.release()
    if shots:
        cv2.imwrite(os.path.join(OUT, tag, key + "_shots.jpg"), np.vstack([shots[k] for k in sorted(shots)]))
    print(f"{key}: {n} frames, proc {1000 * np.mean(t_proc):.1f} ms/frame", flush=True)


if __name__ == "__main__":
    model = V.Model()
    tag = sys.argv[1] if len(sys.argv) > 1 else "v10a"
    keys = sys.argv[2:] or ["2026-09-23_05-46-00_raw", "2026-09-22_23-02-18"]
    for k in keys:
        base = k.replace("_raw", "")
        run(k, tag, lambda: V.Renderer(model), frames=set(range(0, 2000, 37)), compare=os.path.join(OUT, "v9", base + ".mp4"))
