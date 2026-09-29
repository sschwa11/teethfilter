import argparse
import datetime
import json
import os
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "face_landmarker.task")
SETTINGS_FILE = os.path.join(HERE, "settings.json")
RECORD_DIR = os.path.join(HERE, "recordings")
DEFAULTS = {"dark": 0.55, "extent": 0.7, "max_gap": 0.24, "min_gap": 0.025, "mirror_extent": 0.55}
SOURCES = ["self", "reference", "procedural"]
AI_MODEL_PATH = os.path.join(HERE, "teeth_ai.onnx")
AI_CROP = (70, 190, 40, 280)
LEN_CACHE = {}

UPPER_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
LOWER_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
INNER_POLY = UPPER_INNER + LOWER_INNER[::-1][1:-1]
NOSE_TIP = 1

CW, CH = 320, 260
MOUTH_PX = 200.0
AW, AH = 512, 160
VMAX = 0.4

GY, GX = np.mgrid[0:CH, 0:CW].astype(np.float32)
GU = (GX - CW / 2) / (MOUTH_PX / 2)
U = GU[0]
XS = np.arange(CW)


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def mouth_frame(pts):
    a = pts[78].astype(np.float64)
    b = pts[308].astype(np.float64)
    v = b - a
    if v[0] < 0:
        a, b, v = b, a, -v
    w = max(np.linalg.norm(v), 1e-3)
    ex = v / w
    ey = np.array([-ex[1], ex[0]])
    c = (a + b) / 2
    if np.dot(pts[NOSE_TIP] - c, ey) > 0:
        ey = -ey
    s = MOUTH_PX / w
    r = np.stack([ex, ey])
    m = np.zeros((2, 3), np.float64)
    m[:, :2] = s * r
    m[:, 2] = np.array([CW / 2, CH / 2]) - s * r @ c
    return m


def to_canon(m, pts):
    return pts @ m[:, :2].T + m[:, 2]


def lip_curves(cp):
    up = cp[UPPER_INNER]
    lo = cp[LOWER_INNER]
    ou = np.argsort(up[:, 0])
    ol = np.argsort(lo[:, 0])
    yu = np.interp(XS, up[ou, 0], up[ou, 1])
    yl = np.interp(XS, lo[ol, 0], lo[ol, 1])
    return yu.astype(np.float32), yl.astype(np.float32)


def mouth_mask(cp):
    mk = np.zeros((CH, CW), np.uint8)
    cv2.fillPoly(mk, [np.round(cp[INNER_POLY]).astype(np.int32)], 255, lineType=cv2.LINE_AA)
    return mk


def build_procedural(upper):
    if upper:
        widths = [8.5, 6.6, 7.6, 7.0, 6.6, 10.0]
        heights = [10.5, 9.0, 10.0, 8.2, 7.4, 6.8]
        pointy = [0, 0, 1, 0.4, 0.3, 0]
    else:
        widths = [5.4, 5.9, 7.0, 7.0, 7.2, 11.0]
        heights = [7.5, 7.8, 9.0, 7.0, 6.5, 6.0]
        pointy = [0, 0, 0.8, 0.3, 0.2, 0]
    radius, half, mw, gap = 23.0, 25.0, 50.0, 0.25
    u = np.linspace(-1, 1, AW, dtype=np.float32)[None, :]
    v = np.linspace(0, VMAX, AH, dtype=np.float32)[:, None]
    rel = np.full((AH, AW), 0.8, np.float32)
    db = np.zeros((AH, AW), np.float32)
    cov = np.zeros((AH, AW), np.float32)
    for side in (-1, 1):
        s = 0.0
        for w, h, pt in zip(widths, heights, pointy):
            s0, s1 = s, s + w
            s = s1
            x0 = radius * np.sin(min(s0 / radius, np.pi / 2))
            x1 = radius * np.sin(min(s1 / radius, np.pi / 2))
            if x1 - x0 < 0.4:
                break
            lo_u, hi_u = sorted((side * (x0 + gap / 2) / half, side * (x1 - gap / 2) / half))
            cu, hw = (lo_u + hi_u) / 2, (hi_u - lo_u) / 2
            hv = h / mw
            px = np.abs((u - cu) / hw)
            py = v / hv
            f = px ** (4 - 1.8 * pt) + np.clip(py, 0, None) ** (6 - 3 * pt)
            a = np.clip((1 - f) * 4, 0, 1) * (px <= 1.05)
            shade = 1.04 * (1 - 0.12 * px ** 3) * (0.9 + 0.1 * smoothstep(0, 0.3, py))
            shade *= 1 - 0.06 * smoothstep(0.75, 1.0, py)
            shade *= 1 - 0.3 * np.clip(np.abs(u), 0, 1) ** 2.5
            yel = 4 * (1 - smoothstep(0.0, 0.5, py)) - 3 * smoothstep(0.7, 1.0, py)
            upd = a > cov
            rel = np.where(upd, shade * a + rel * (1 - a), rel)
            db = np.where(upd, yel * a, db)
            cov = np.maximum(cov, a)
    rel = cv2.GaussianBlur(rel, (0, 0), 1.5)
    env = cv2.dilate(cov, np.ones((1, 9), np.uint8))
    env = cv2.GaussianBlur(env, (0, 0), 1.2)
    da = np.zeros_like(db)
    return np.dstack([rel, da, db, np.clip(env, 0, 1)]).astype(np.float32)


def build_reference(img, pts):
    m = mouth_frame(pts)
    crop = cv2.warpAffine(img, m.astype(np.float32), (CW, CH), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    cp = to_canon(m, pts)
    yu, yl = lip_curves(cp)
    mk = mouth_mask(cp)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    xs = XS.astype(np.float32)
    au = np.linspace(-1, 1, AW, dtype=np.float32)
    av = np.linspace(0, VMAX, AH, dtype=np.float32)
    mapx = np.tile((CW / 2 + au * MOUTH_PX / 2)[None, :], (AH, 1)).astype(np.float32)
    out = []
    for upper in (True, False):
        base = np.interp(mapx[0], xs, yu if upper else yl).astype(np.float32)
        sign = 1 if upper else -1
        mapy = (base[None, :] + sign * av[:, None] * MOUTH_PX).astype(np.float32)
        samp = cv2.remap(lab, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        inm = cv2.remap(mk, mapx, mapy, cv2.INTER_LINEAR, borderValue=0) > 127
        lch = samp[..., 0]
        if inm.sum() < 50:
            out.append(None)
            continue
        thr = 0.7 * np.percentile(lch[inm], 92)
        bright = (inm & (lch > thr)).astype(np.uint8)
        bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((5, 9), np.uint8))
        rows = np.arange(AH)[:, None]
        ext = np.where(bright.any(0), (bright * rows).max(0), 0).astype(np.float32)
        ext = cv2.medianBlur(ext.reshape(1, -1).astype(np.float32), 5).ravel()
        env = ((rows <= ext[None, :]) & inm).astype(np.float32)
        env = cv2.GaussianBlur(env, (0, 0), 1.2)
        sel = bright > 0
        if sel.sum() < 30:
            out.append(None)
            continue
        ml, ma, mb = [samp[..., i][sel].mean() for i in range(3)]
        out.append(np.dstack([lch / max(ml, 1), samp[..., 1] - ma, samp[..., 2] - mb, env]).astype(np.float32))
    return out[0], out[1]


def atlas_center_len(atlas):
    key = id(atlas)
    if key not in LEN_CACHE:
        env = atlas[..., 3] > 0.5
        cols = np.abs(np.linspace(-1, 1, AW)) < 0.3
        rows = np.arange(AH)[:, None]
        ext = (env[:, cols] * rows).max(0)
        LEN_CACHE[key] = max(float(np.median(ext)) / (AH - 1) * VMAX, 0.02)
    return LEN_CACHE[key]


class State:
    def __init__(self, priors=None):
        self.pts = None
        self.alpha = None
        self.rep = None
        self.color = np.array([190.0, 130.0, 136.0])
        self.fade = 0.0
        self.motion = 0.0
        self.otsu = None
        self.otsu_base = None
        self.miss = {}
        self.prior = priors if priors is not None else {}
        self.calib_frames = 0
        self.gaps = []
        self.tracks = []
        self.mcols = None
        self.edge_mem = dict(self.prior.pop("_edge")) if "_edge" in self.prior else None
        self.cax = None
        self.c_mask = None
        self.c_last = None
        self.c_miss = 0
        self.prior_ready = bool(priors)


class Config:
    def __init__(self, a):
        self.dark = a.dark
        self.extent = a.extent
        self.max_gap = a.max_gap
        self.mirror_extent = a.mirror_extent
        self.mirror = True
        self.method = getattr(a, "method", "edge")
        self.ai = None
        self.fill = "mirror"
        self.min_gap = a.min_gap
        self.min_open = a.min_open
        self.upper = not a.no_upper
        self.lower = not a.no_lower
        self.enabled = True
        self.debug = False
        self.hud = False
        self.source = "self"
        self.auto = a.auto
        self.camera_name = ""
        self.hotkey = ""


def smooth_points(st, pts):
    if st.pts is None or st.pts.shape != pts.shape:
        st.pts = pts.copy()
        st.motion = 0.0
    else:
        d = float(np.mean(np.abs(pts - st.pts)))
        st.motion = d
        k = float(np.clip(0.45 + d / 5.0, 0.45, 1.0))
        st.pts = k * pts + (1 - k) * st.pts
    return st.pts


def load_settings():
    try:
        with open(SETTINGS_FILE, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}, {}
    vals = {}
    for k in DEFAULTS:
        try:
            vals[k] = float(data[k])
        except (KeyError, TypeError, ValueError):
            pass
    priors = {}
    em = data.get("edge_memory")
    if isinstance(em, dict) and all(k in em for k in ("u", "side", "tw")):
        try:
            priors["_edge"] = {"u": float(em["u"]), "side": int(em["side"]), "tw": float(em["tw"]), "conf": 8}
        except (TypeError, ValueError):
            pass
    ao = data.get("anchor_offset")
    if isinstance(ao, list) and len(ao) == 2:
        priors["_anchor"] = [float(v) for v in ao]
    for k in ("upper", "lower"):
        arr = data.get("missing_" + k)
        if isinstance(arr, list) and len(arr) == CW:
            priors[k] = np.clip(np.array(arr, np.float32), 0, 1)
    return vals, priors


def save_settings(cfg, st, anchor=None):
    data = {k: round(float(getattr(cfg, k)), 3) for k in DEFAULTS}
    if anchor is not None and anchor.et.n >= 15:
        data["anchor_offset"] = [round(float(v), 2) for v in anchor.offset]
        anchor.save_state()
    elif "_anchor" in st.prior:
        data["anchor_offset"] = st.prior["_anchor"]
    for k, arr in st.prior.items():
        if k.startswith("_"):
            continue
        data["missing_" + k] = [round(float(x), 2) for x in arr]
    if st.edge_mem is not None and st.edge_mem["conf"] >= 6:
        data["edge_memory"] = {"u": round(st.edge_mem["u"], 4), "side": st.edge_mem["side"], "tw": round(st.edge_mem["tw"], 2)}
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except OSError:
        pass
    return {k: data[k] for k in DEFAULTS}


def otsu_split(vals):
    if vals.size < 80:
        return None
    v8 = np.clip(vals, 0, 255).astype(np.uint8).reshape(-1, 1)
    t, _ = cv2.threshold(v8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    lo = vals[vals <= t]
    hi = vals[vals > t]
    if lo.size < 0.04 * vals.size or hi.size < 0.1 * vals.size:
        return None
    if hi.mean() - lo.mean() < 25:
        return None
    return float(t)


def segments(mask):
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def find_gaps(bright, inside, yu, yl, cfg, st):
    k = max(5, int(cfg.max_gap * MOUTH_PX))
    b8 = bright.astype(np.uint8)
    closed = cv2.morphologyEx(b8, cv2.MORPH_CLOSE, np.ones((1, k), np.uint8)) > 0
    zone = inside & (np.abs(GU) < cfg.extent)
    gap = (closed & ~bright & zone).astype(np.uint8)
    gap = cv2.morphologyEx(gap, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(gap, connectivity=8)
    minw = cfg.min_gap * MOUTH_PX
    near = 0.1 * MOUTH_PX
    found = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if w < minw or h < 0.025 * MOUTH_PX or area < 0.4 * minw * minw:
            continue
        if h < 0.25 * w:
            continue
        if max(abs(U[x]), abs(U[min(x + w, CW - 1)])) > cfg.extent:
            continue
        cxi = int(x + w // 2)
        touches_up = (y - yu[cxi]) < near
        touches_lo = (yl[cxi] - (y + h)) < near
        if not (touches_up or touches_lo):
            continue
        upper = touches_up if touches_up != touches_lo else (y + h / 2) < (yu[cxi] + yl[cxi]) / 2
        key = "upper" if upper else "lower"
        pr = st.prior.get(key)
        if pr is not None and st.prior_ready:
            pm = float(pr[x:x + w].mean())
            if pm < 0.2:
                continue
        found.append({"mask": labels == i, "x0": int(x), "x1": int(x + w), "y0": int(y), "y1": int(y + h), "upper": bool(upper), "key": key})
    return found


def fill_gap(g, crop, lab, bright, yu, yl, taken, st):
    a, b, y0, y1 = g["x0"], g["x1"], g["y0"], g["y1"]
    w = b - a
    lip = yu if g["upper"] else yl
    pad = 3
    seg = np.arange(max(0, a - pad), min(CW, b + pad))
    rows = slice(max(0, y0 - 2), min(CH, y1 + 2))
    donor = None
    options = (CW - 1 - seg,) if g.get("mirror") else (CW - 1 - seg, seg - w - 1, seg + w + 1)
    for cand in options:
        if cand.min() < 0 or cand.max() >= CW:
            continue
        if taken[cand].any() and not g.get("mirror"):
            continue
        frac = bright[rows][:, cand].mean()
        if g.get("mirror") or frac > 0.55:
            donor = cand
            break
    rep = crop.astype(np.float32).copy()
    if donor is not None:
        mapx = np.broadcast_to(donor[None, :].astype(np.float32), (CH, seg.size))
        mapy = (GY[:, seg] + (lip[donor] - lip[seg])[None, :]).astype(np.float32)
        patch = cv2.remap(crop, np.ascontiguousarray(mapx), mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
        tgt = np.zeros((CH, CW), bool)
        tgt[rows, max(0, a - 14):a] = True
        tgt[rows, b:b + 14] = True
        tgt &= bright
        plab = cv2.cvtColor(np.clip(patch, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
        src = plab[rows, :, 0]
        src = src[src > np.percentile(src, 40)] if src.size else src
        if tgt.sum() > 15 and src.size > 15:
            gain = float(np.clip(lab[..., 0][tgt].mean() / max(src.mean(), 1), 0.8, 1.25))
            plab[..., 0] *= gain
            patch = cv2.cvtColor(np.clip(plab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)
        rep[:, seg] = patch
    else:
        la = max(0, a - 3)
        rb = min(CW - 1, b + 2)
        left = crop[:, la].astype(np.float32)
        right = crop[:, rb].astype(np.float32)
        t = ((seg - la) / max(rb - la, 1)).astype(np.float32)
        rep[:, seg] = left[:, None, :] * (1 - t[None, :, None]) + right[:, None, :] * t[None, :, None]
        edge = np.exp(-((seg - a) / 1.2) ** 2) + np.exp(-((seg - b) / 1.2) ** 2)
        rep[:, seg] *= (1 - 0.18 * np.clip(edge, 0, 1))[None, :, None]
    k = (5, 3) if g.get("mirror") else (5, 5)
    mask = cv2.dilate((g["mask"] > 0.35).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, k)).astype(np.float32)
    return rep, mask, donor is not None


def fill_gap_atlas(g, atlas, yu, yl, st):
    lip = yu if g["upper"] else yl
    depth = (GY - lip[None, :]) / MOUTH_PX if g["upper"] else (lip[None, :] - GY) / MOUTH_PX
    e = max((g["y1"] - lip[(g["x0"] + g["x1"]) // 2]) / MOUTH_PX if g["upper"] else (lip[(g["x0"] + g["x1"]) // 2] - g["y0"]) / MOUTH_PX, 0.03)
    ay = (depth * (AH - 1) / VMAX * atlas_center_len(atlas) / e).astype(np.float32)
    ax = ((GU + 1) * 0.5 * (AW - 1)).astype(np.float32)
    at = cv2.remap(atlas, ax, ay, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    tl, ta, tb = st.color
    alab = np.dstack([tl * at[..., 0], ta + at[..., 1], tb + at[..., 2]])
    rep = cv2.cvtColor(np.clip(alab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)
    mask = cv2.dilate((g["mask"] > 0.35).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))).astype(np.float32)
    return rep, mask


def find_mirror(bright, inside, yu, yl, cfg, st):
    mid = (yu + yl) / 2
    band = inside & (GY >= yu[None, :] - 2) & (GY <= mid[None, :] + 0.02 * MOUTH_PX)
    cnt = band.sum(0)
    bb = bright & band
    n = bb.sum(0).astype(np.float32)
    n = np.convolve(n, np.ones(3, np.float32) / 3, mode="same")
    hasb = bb.any(0)
    firstb = np.where(hasb, bb.argmax(0), CH)
    hangs = firstb <= yu + 0.4 * np.maximum(yl - yu, 1)
    nm = np.where(hangs, n, 0)[::-1]
    lo_t = 0.018 * MOUTH_PX
    hi_t = 0.045 * MOUTH_PX
    zone = (np.abs(U) < cfg.mirror_extent) & (cnt > 4)
    raw = zone & (n < lo_t) & (nm > hi_t)
    raw = cv2.morphologyEx(raw.astype(np.uint8).reshape(1, -1), cv2.MORPH_CLOSE, np.ones((1, 5), np.uint8)).ravel() > 0
    raw &= zone & (nm > hi_t * 0.7)
    if st.mcols is None:
        st.mcols = np.zeros(CW, np.float32)
    seen = zone & (nm > hi_t * 0.6)
    st.mcols = np.where(seen, 0.85 * st.mcols + 0.15 * raw, st.mcols)
    memory = st.mcols > 0.2
    pr = st.prior.get("upper")
    if pr is not None and st.prior_ready:
        memory |= pr > 0.5
    cand = raw | (memory & (n < lo_t * 1.5))
    cand &= zone & (nm > hi_t * 0.6)
    cand = cv2.morphologyEx(cand.astype(np.uint8).reshape(1, -1), cv2.MORPH_CLOSE, np.ones((1, 11), np.uint8)).ravel() > 0
    cand &= zone & (n < hi_t * 0.6)
    out = []
    minw = cfg.min_gap * MOUTH_PX
    has = bb.any(0)
    first = np.where(has, bb.argmax(0), 0)
    last = np.where(has, CH - 1 - bb[::-1].argmax(0), 0)
    for a, b in segments(cand):
        if b - a < minw:
            continue
        seg = np.arange(a, b)
        xm = CW - 1 - seg
        dy = yu[seg] - yu[xm]
        ok = has[xm] & (first[xm] <= yu[xm] + 0.4 * np.maximum(yl[xm] - yu[xm], 1))
        if ok.sum() < max(2, 0.4 * seg.size):
            continue
        top = (first[xm] + dy).astype(np.float32)
        bot = (last[xm] + dy).astype(np.float32)
        idx = np.arange(seg.size)
        top = np.interp(idx, idx[ok], top[ok]).astype(np.float32)
        bot = np.interp(idx, idx[ok], bot[ok]).astype(np.float32)
        kw = min(9, seg.size - (1 - seg.size % 2))
        if kw >= 3:
            top = cv2.medianBlur(top.reshape(1, -1), 5 if kw >= 5 else 3).ravel()
            bot = cv2.medianBlur(bot.reshape(1, -1), 5 if kw >= 5 else 3).ravel()
        top = np.minimum(top, np.percentile(top, 50))
        bot = np.maximum(bot, np.percentile(bot, 50))
        top = np.clip(top, 0, CH - 1)
        bot = np.clip(bot, 0, CH - 1)
        mask = np.zeros((CH, CW), bool)
        rows = GY[:, seg]
        mask[:, seg] = (rows >= top[None, :]) & (rows <= bot[None, :])
        mask &= inside | (GY <= mid[None, :])
        if mask.sum() < minw * 3:
            continue
        ys = np.flatnonzero(mask.any(1))
        out.append({"mask": mask, "x0": int(a), "x1": int(b), "y0": int(ys.min()), "y1": int(ys.max()) + 1, "upper": True, "key": "upper", "mirror": True})
    return out


def column_fill(mask):
    has = mask.any(0)
    if not has.any():
        return mask
    top = np.where(has, mask.argmax(0), CH)
    bot = np.where(has, CH - 1 - mask[::-1].argmax(0), -1)
    rows = np.arange(CH)[:, None]
    return (rows >= top[None, :]) & (rows <= bot[None, :])


def valid_donor(donor, yu, yl, upper):
    has = donor.any(0)
    if upper:
        bot = np.where(has, CH - 1 - donor[::-1].argmax(0), -1)
        ok = bot > yu + 2
    else:
        top = np.where(has, donor.argmax(0), CH)
        ok = top < yl - 2
    return donor & ok[None, :]


def find_edge_mirror(bright, inside, yu, yl, cfg):
    open_ = np.maximum(yl - yu, 1)
    out = []
    minw = cfg.min_gap * MOUTH_PX
    maxtw = 0.17 * MOUTH_PX
    for upper in (True, False):
        if upper:
            region = inside & (GY <= np.maximum(yu + 0.65 * open_, yu + 0.12 * MOUTH_PX)[None, :])
        else:
            region = inside & (GY >= np.minimum(yl - 0.65 * open_, yl - 0.12 * MOUTH_PX)[None, :])
        b = cv2.morphologyEx((bright & region).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
        for i in range(1, n):
            x, y, w, h, area = [int(v) for v in stats[i]]
            if area < 0.0015 * MOUTH_PX * MOUTH_PX or h < 0.03 * MOUTH_PX or w < minw:
                continue
            cxi = min(max(x + w // 2, 0), CW - 1)
            if upper and y > yu[cxi] + 0.4 * open_[cxi]:
                continue
            if upper and y + h < yu[cxi] + 3:
                continue
            if not upper and y + h < yl[cxi] - 0.4 * open_[cxi]:
                continue
            comp = labels == i
            r0, r1 = y + h // 4, max(y + 3 * h // 4, y + h // 4 + 1)
            for side in (-1, 1):
                edge = x if side == -1 else x + w - 1
                k = 0
                bounded = False
                while True:
                    xx = edge + side * (k + 1)
                    if xx < 0 or xx >= CW:
                        break
                    if inside[r0:r1, xx].mean() < 0.5:
                        break
                    if bright[r0:r1, xx].mean() > 0.3:
                        bounded = True
                        break
                    k += 1
                tw = int(min(w, maxtw))
                if k < max(0.45 * tw, minw):
                    continue
                gc = edge + side * (min(tw, k) // 2)
                if abs(U[min(max(gc, 0), CW - 1)]) > min(cfg.mirror_extent, 0.4):
                    continue
                if not bounded and not (upper and abs(U[edge]) < 0.32):
                    continue
                if bounded and k > 2.2 * tw:
                    continue
                fw = min(tw, k + 2)
                if side == 1:
                    tcols = np.arange(edge + 1, min(CW, edge + 1 + fw))
                else:
                    tcols = np.arange(max(0, edge - fw), edge)
                dcols = 2 * edge - tcols
                okc = (dcols >= 0) & (dcols < CW)
                tcols, dcols = tcols[okc], dcols[okc]
                if tcols.size < minw:
                    continue
                donor = column_fill(cv2.morphologyEx(comp.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 3), np.uint8)) > 0)
                donor = valid_donor(donor, yu, yl, upper)
                mask = np.zeros((CH, CW), bool)
                mask[:, tcols] = donor[:, dcols]
                mask &= inside | (np.abs(GY - (yu if upper else yl)[None, :]) < 3)
                if mask.sum() < minw * 4:
                    continue
                ys = np.flatnonzero(mask.any(1))
                if not upper and (ys.max() - ys.min()) < 0.05 * MOUTH_PX:
                    continue
                out.append({"mask": mask, "x0": int(tcols.min()), "x1": int(tcols.max()) + 1, "y0": int(ys.min()), "y1": int(ys.max()) + 1,
                            "upper": upper, "key": "upper" if upper else "lower", "edge": True, "axis": float(edge)})
    return out


def upward(mask, px):
    out = mask.copy()
    for k in range(1, px + 1):
        out[:-k] |= mask[k:]
    return out


def update_edge_memory(st, gaps):
    mem = st.edge_mem
    for g in gaps:
        if not g.get("edge") or not g["upper"]:
            continue
        side = 1 if g["x0"] > g["axis"] else -1
        ua = float(U[int(np.clip(round(g["axis"]), 0, CW - 1))])
        if abs(ua) > 0.3:
            continue
        tw = g["x1"] - g["x0"]
        if mem is not None and mem["side"] == side and abs(mem["u"] - ua) < 0.1:
            mem["u"] = 0.8 * mem["u"] + 0.2 * ua
            mem["tw"] = 0.8 * mem["tw"] + 0.2 * tw
            mem["conf"] = min(mem["conf"] + 1, 90)
        elif mem is None or mem["conf"] < 5:
            mem = {"u": ua, "side": side, "tw": float(tw), "conf": 1}
        else:
            mem["conf"] -= 1
    st.edge_mem = mem


def edge_from_memory(bright, region_in, yu, yl, st):
    mem = st.edge_mem
    if mem is None or mem["conf"] < 6:
        return None
    open_ = np.maximum(yl - yu, 1)
    ax = st.cax if st.cax is not None else CW / 2 + mem["u"] * MOUTH_PX / 2
    side = mem["side"]
    tooth_dir = -side
    band = region_in & (GY >= (yu - 0.06 * MOUTH_PX)[None, :]) & (GY <= (yu + np.minimum(0.5 * open_, 0.1 * MOUTH_PX))[None, :])
    frac = np.where(band.sum(0) > 2, (bright & band).sum(0) / np.maximum(band.sum(0), 1), 0)
    lo = int(max(1, ax - 0.07 * MOUTH_PX))
    hi = int(min(CW - 2, ax + 0.07 * MOUTH_PX))
    edge = None
    cols = range(lo, hi + 1) if tooth_dir == 1 else range(hi, lo - 1, -1)
    for x in cols:
        run = [x + tooth_dir * j for j in range(6)]
        if all(0 <= c < CW and frac[c] > 0.45 for c in run):
            edge = x
            break
    if edge is None:
        return None
    st.cax = float(edge) if st.cax is None else 0.7 * st.cax + 0.3 * float(edge)
    b = (bright & band).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
    rows = np.flatnonzero(b[:, edge])
    if rows.size == 0:
        return None
    lab_id = labels[rows[rows.size // 2], edge]
    if lab_id == 0:
        return None
    comp = labels == lab_id
    edge = int(round(st.cax))
    tw = int(max(mem["tw"], 0.06 * MOUTH_PX))
    if side == 1:
        tcols = np.arange(edge + 1, min(CW, edge + 1 + tw))
    else:
        tcols = np.arange(max(0, edge - tw), edge)
    dcols = 2 * edge - tcols
    okc = (dcols >= 0) & (dcols < CW)
    tcols, dcols = tcols[okc], dcols[okc]
    if tcols.size < 4:
        return None
    donor = column_fill(cv2.morphologyEx(comp.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 3), np.uint8)) > 0)
    donor = valid_donor(donor, yu, yl, True)
    mask = np.zeros((CH, CW), bool)
    mask[:, tcols] = donor[:, dcols]
    mask &= region_in
    if mask.sum() < 30:
        return None
    ys = np.flatnonzero(mask.any(1))
    return {"mask": mask, "x0": int(tcols.min()), "x1": int(tcols.max()) + 1, "y0": int(ys.min()), "y1": int(ys.max()) + 1,
            "upper": True, "key": "upper", "edge": True, "axis": float(edge)}


class AIFill:
    def __init__(self, path):
        import onnxruntime as ort
        avail = ort.get_available_providers()
        providers = [p for p in ("DmlExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider") if p in avail]
        self.sess = ort.InferenceSession(path, providers=providers)
        self.name = self.sess.get_inputs()[0].name
        self.provider = self.sess.get_providers()[0]
        shape = self.sess.get_inputs()[0].shape
        self.cin = shape[1] if isinstance(shape[1], int) else 7

    def run(self, crop, mask, hint=None):
        y0, y1, x0, x1 = AI_CROP
        img = crop[y0:y1, x0:x1].astype(np.float32) / 255
        m = mask[y0:y1, x0:x1].astype(np.float32)[..., None]
        parts = [img * (1 - m) + 0.5 * m, m]
        if self.cin == 7:
            h = (hint if hint is not None else crop)[y0:y1, x0:x1].astype(np.float32) / 255
            parts.append(h * m + 0.5 * (1 - m))
        x = np.concatenate(parts, 2).transpose(2, 0, 1)[None].astype(np.float32)
        out = self.sess.run(None, {self.name: x})[0][0].transpose(1, 2, 0)
        rep = crop.astype(np.float32).copy()
        rep[y0:y1, x0:x1] = np.clip(out * 255, 0, 255)
        return rep


def load_ai(path=AI_MODEL_PATH):
    if not os.path.exists(path):
        return None
    try:
        return AIFill(path)
    except Exception as e:
        print("AI model not loaded:", e)
        return None


def fill_edge(g, crop):
    mapx = (2 * g["axis"] - GX).astype(np.float32)
    rep = cv2.remap(crop, mapx, GY, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
    m = (g["mask"] > 0.35).astype(np.float32)
    m = cv2.GaussianBlur(m, (0, 0), 0.8)
    return rep, m


def lip_free_bright(lab, inside, thr, yu, yl):
    lch, ach = lab[..., 0], lab[..., 1]
    bright = inside & (lch >= thr)
    cx = slice(int(CW / 2 - 0.25 * MOUTH_PX), int(CW / 2 + 0.25 * MOUTH_PX))
    lip_rows = (GY > yl[None, :] + 0.03 * MOUTH_PX) & (GY < yl[None, :] + 0.09 * MOUTH_PX)
    lip_rows[:, :cx.start] = False
    lip_rows[:, cx.stop:] = False
    lip_a = ach[lip_rows]
    core = GY >= yu[None, :]
    top = bright & core & (GY < ((yu + yl) / 2)[None, :])
    if top.sum() > 30:
        ta = float(np.percentile(ach[top], 30))
        if lip_a.size > 50:
            la = float(np.median(lip_a))
            if la - ta > 3:
                bright &= ach < (la + ta) / 2
        bright &= core | (ach < ta + 3)
    else:
        bright &= core
    return cv2.morphologyEx(bright.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 1), np.uint8)) > 0


def update_tracks(st, gaps):
    for tr in st.tracks:
        tr["matched"] = False
    fresh = []
    for g in gaps:
        best, score = None, 0.0
        for tr in st.tracks:
            if tr["key"] != g["key"] or tr["matched"]:
                continue
            ov = min(tr["x1"], g["x1"]) - max(tr["x0"], g["x0"])
            un = max(tr["x1"], g["x1"]) - min(tr["x0"], g["x0"])
            o = ov / max(un, 1)
            if o > score:
                best, score = tr, o
        if best is not None and score > 0.25:
            best["mask"] = 0.5 * best["mask"] + 0.5 * g["mask"].astype(np.float32)
            for k in ("x0", "x1", "y0", "y1", "upper"):
                best[k] = g[k]
            best["mirror"] = g.get("mirror", False)
            if "axis" in g:
                best["axis"] = g["axis"]
                best["edge"] = True
            best["hits"] += 1
            best["ttl"] = 15
            best["matched"] = True
        else:
            t = dict(g)
            t.update(mask=g["mask"].astype(np.float32), hits=1, ttl=15, matched=True)
            fresh.append(t)
    for tr in st.tracks:
        if not tr["matched"]:
            tr["ttl"] -= 1
            tr["mask"] = tr["mask"] * 0.9
    tracks = sorted([t for t in st.tracks if t["ttl"] > 0] + fresh, key=lambda t: (not t.get("mirror"), -t["hits"]))
    merged = []
    for t in tracks:
        host = None
        for m in merged:
            if m["key"] == t["key"] and min(m["x1"], t["x1"]) > max(m["x0"], t["x0"]):
                host = m
                break
        if host is None:
            merged.append(t)
            continue
        if t.get("edge") or host.get("edge"):
            continue
        if bool(t.get("mirror")) == bool(host.get("mirror")):
            host["mask"] = np.maximum(host["mask"], t["mask"])
            host["x0"] = min(host["x0"], t["x0"])
            host["x1"] = max(host["x1"], t["x1"])
            host["y0"] = min(host["y0"], t["y0"])
            host["y1"] = max(host["y1"], t["y1"])
            host["hits"] = max(host["hits"], t["hits"])
            host["ttl"] = max(host["ttl"], t["ttl"])
    st.tracks = merged
    return [t for t in st.tracks if t["hits"] >= 2]


def calibrate(cfg, st, bright, inside, gaps, yu, yl):
    if st.motion > 3.0 or st.fade < 0.9:
        return
    for key, lip, sign in (("upper", yu, 1), ("lower", yl, -1)):
        depth = (GY - lip[None, :]) * sign / MOUTH_PX
        band = inside & (depth > -0.02) & (depth < 0.12)
        seen = band.sum(0) > 4
        cur = np.zeros(CW, np.float32)
        for g in gaps:
            if g["key"] == key:
                cur[g["x0"]:g["x1"]] = 1
        pr = st.prior.get(key)
        if pr is None:
            pr = np.full(CW, 0.5, np.float32)
        pr = np.where(seen, pr + 0.03 * (cur - pr), pr)
        st.prior[key] = pr.astype(np.float32)
    teeth_u = np.abs(GU[bright & inside])
    if teeth_u.size > 150:
        reach = float(np.percentile(teeth_u, 97)) + 0.05
        cfg.extent += 0.04 * (float(np.clip(reach, 0.55, 0.95)) - cfg.extent)
    st.calib_frames += 1
    if st.calib_frames >= 150:
        st.prior_ready = True


def filter_frame(frame, pts, st, cfg, atlases, anchor):
    if anchor is not None and cfg.method == "anchor":
        anchor.upper, anchor.lower = cfg.upper, cfg.lower
        return anchor.process(frame, pts)
    return process(frame, smooth_points(st, pts[:, :2]), st, cfg, atlases)


def make_anchor(cfg, priors):
    if cfg.method != "anchor":
        return None
    import dental
    if not dental.available():
        print("teeth_model.npz / teeth_seg.onnx not found; falling back to --method edge.")
        cfg.method = "edge"
        return None
    anchor = dental.AnchorFill(offset=priors.get("_anchor"))
    loaded = anchor.load_state()
    print(f"Anchor fill ready (mouth segmenter on {anchor.seg.provider}){', tooth memory loaded' if loaded else ''}.")
    return anchor


def process(frame, pts, st, cfg, atlases):
    m = mouth_frame(pts)
    cp = to_canon(m, pts)
    yu, yl = lip_curves(cp)
    cx = slice(int(CW / 2 - MOUTH_PX * 0.2), int(CW / 2 + MOUTH_PX * 0.2))
    open_ratio = float(np.median((yl - yu)[cx])) / MOUTH_PX
    target = float(smoothstep(cfg.min_open, cfg.min_open + 0.025, open_ratio))
    st.fade = 0.3 * st.fade + 0.7 * target
    if st.fade < 0.02:
        st.alpha = None
        st.rep = None
        st.tracks = []
        return frame, None
    crop = cv2.warpAffine(frame, m.astype(np.float32), (CW, CH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    lch = lab[..., 0]
    inside = cv2.erode(mouth_mask(cp), np.ones((3, 3), np.uint8)) > 127
    if inside.sum() < 60:
        return frame, None
    zone = inside & (np.abs(GU) < 0.85)
    t = otsu_split(lch[zone])
    r0 = int(np.clip(np.median(yu[cx]) - 0.34 * MOUTH_PX, 0, CH - 2))
    r1 = int(np.clip(np.median(yu[cx]) - 0.2 * MOUTH_PX, r0 + 1, CH - 1))
    skin_l = float(np.median(lch[r0:r1, cx]))
    p90 = float(np.percentile(lch[zone], 90)) if zone.any() else skin_l
    st.otsu_base = max(p90, skin_l)
    fallback_t = cfg.dark * st.otsu_base
    if t is not None:
        st.otsu = t if st.otsu is None else 0.8 * st.otsu + 0.2 * t
    thr = st.otsu if st.otsu is not None else fallback_t
    dark = lch < thr
    inside_up = upward(inside, int(0.05 * MOUTH_PX))
    bright = lip_free_bright(lab, inside_up, thr, yu, yl)
    chroma = np.hypot(lab[..., 1] - 128, lab[..., 2] - 128)
    real = bright & (lch > thr * 1.15) & (chroma < 30)
    if real.sum() > 40:
        mean = np.array([lab[..., i][real].mean() for i in range(3)])
        st.color = 0.85 * st.color + 0.15 * mean
    if cfg.method == "edge":
        raw = find_edge_mirror(bright, inside_up, yu, yl, cfg)
        update_edge_memory(st, raw)
        central = None
        if cfg.upper and st.edge_mem is not None and st.edge_mem["conf"] >= 6:
            central = edge_from_memory(bright, inside_up, yu, yl, st)
            ca = st.cax if st.cax is not None else CW / 2 + st.edge_mem["u"] * MOUTH_PX / 2
            if central is None:
                near = [g for g in raw if g["upper"] and abs(g["axis"] - ca) < 0.08 * MOUTH_PX and (1 if g["x0"] > g["axis"] else -1) == st.edge_mem["side"]]
                if near:
                    central = max(near, key=lambda g: g["mask"].sum())
                    st.cax = 0.7 * ca + 0.3 * central["axis"]
            if central is not None:
                cm = central["mask"].astype(np.float32)
                if st.c_mask is not None and st.c_miss == 0:
                    cm = 0.55 * st.c_mask + 0.45 * cm
                central["mask"] = cm
                st.c_mask, st.c_last, st.c_miss = cm, central, 0
            elif st.c_last is not None and st.c_miss < 8:
                st.c_miss += 1
                central = dict(st.c_last)
                central["mask"] = st.c_mask * (0.85 ** st.c_miss)
            raw = [g for g in raw if not (g["upper"] and abs(g["axis"] - ca) < 0.12 * MOUTH_PX)]
        gaps = raw
    else:
        gaps = find_gaps(bright, inside, yu, yl, cfg, st)
    if cfg.method != "edge" and cfg.mirror and cfg.upper:
        mirrored = find_mirror(bright, inside, yu, yl, cfg, st)
        for mg in mirrored:
            gaps = [g for g in gaps if not (g["key"] == "upper" and min(g["x1"], mg["x1"]) > max(g["x0"], mg["x0"]))]
        gaps += mirrored
    gaps = [g for g in gaps if (cfg.upper if g["upper"] else cfg.lower)]
    if cfg.auto:
        calibrate(cfg, st, bright, inside, gaps, yu, yl)
    gaps = update_tracks(st, gaps)
    if cfg.method == "edge":
        gaps = [g for g in gaps if g["hits"] >= 3]
        if central is not None:
            gaps = [g for g in gaps if not (g["key"] == "upper" and min(g["x1"], central["x1"]) > max(g["x0"], central["x0"]))] + [central]
    st.gaps = gaps
    taken = np.zeros(CW, bool)
    for g in gaps:
        taken[g["x0"]:g["x1"]] = True
    a = np.zeros((CH, CW), np.float32)
    rep = crop.astype(np.float32).copy()
    ai = getattr(cfg, "ai", None) if getattr(cfg, "fill", "mirror") == "ai" else None
    if ai is not None and gaps:
        hard = np.zeros((CH, CW), bool)
        hint = crop.astype(np.float32).copy()
        for g in gaps:
            sel = g["mask"] > 0.35
            if g.get("edge"):
                rp, _ = fill_edge(g, crop)
                hint[sel] = rp[sel]
            hard |= sel
        hard = cv2.dilate(hard.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))) > 0
        y0, y1, x0, x1 = AI_CROP
        clip = np.zeros_like(hard)
        clip[y0 + 2:y1 - 2, x0 + 2:x1 - 2] = True
        hard &= clip
        if hard.any():
            rep = ai.run(crop, hard, hint)
            a = cv2.GaussianBlur(hard.astype(np.float32), (0, 0), 0.8)
        gaps_to_fill = []
    else:
        gaps_to_fill = gaps
    for g in gaps_to_fill:
        if g.get("edge"):
            rp, mk = fill_edge(g, crop)
        elif cfg.source == "self":
            rp, mk, _ = fill_gap(g, crop, lab, bright, yu, yl, taken, st)
        else:
            rp, mk = fill_gap_atlas(g, atlases[0] if g["upper"] else atlases[1], yu, yl, st)
        take = mk > a
        rep[take] = rp[take]
        a = np.maximum(a, mk)
    mir = [g for g in gaps if g.get("mirror")]
    if mir:
        comp = crop.astype(np.float32) * (1 - a[..., None]) + rep * a[..., None]
        clab = cv2.cvtColor(np.clip(comp, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB)
        cb = ((clab[..., 0] >= thr) & inside).astype(np.uint8)
        zone = np.zeros((CH, CW), bool)
        for g in mir:
            zone[max(0, g["y0"]):g["y1"], max(0, g["x0"] - 14):min(CW, g["x1"] + 14)] = True
        closed = cv2.morphologyEx(cb, cv2.MORPH_CLOSE, np.ones((1, 13), np.uint8)) > 0
        bridge = (closed & (cb == 0) & zone & inside).astype(np.uint8)
        if bridge.any():
            fixed = cv2.inpaint(np.clip(comp, 0, 255).astype(np.uint8), bridge, 3, cv2.INPAINT_TELEA).astype(np.float32)
            bm = cv2.dilate(bridge, np.ones((1, 3), np.uint8)) > 0
            rep[bm] = fixed[bm]
            a[bm] = 1.0
    a *= (inside_up if cfg.method == "edge" else inside).astype(np.float32)
    a = cv2.GaussianBlur(a, (0, 0), 1.0)
    region = (inside_up if cfg.method == "edge" else inside).astype(np.float32)
    if st.alpha is not None and st.alpha.shape == a.shape:
        a = np.where(a > st.alpha, st.alpha + 0.5 * (a - st.alpha), st.alpha + 0.3 * (a - st.alpha))
        a *= region
        if st.rep is not None:
            k = 0.5 if st.motion < 2.0 else 0.2
            both = ((st.alpha > 0.05) & (a > 0.05))[..., None]
            rep = np.where(both, k * st.rep + (1 - k) * rep, rep)
    st.alpha = a
    st.rep = rep
    a = a * st.fade
    if a.max() < 0.01:
        return frame, None
    poly = pts[INNER_POLY]
    mw = np.linalg.norm(pts[308] - pts[78])
    x0 = int(max(poly[:, 0].min() - 0.2 * mw, 0))
    y0 = int(max(poly[:, 1].min() - 0.2 * mw, 0))
    x1 = int(min(poly[:, 0].max() + 0.2 * mw, frame.shape[1]))
    y1 = int(min(poly[:, 1].max() + 0.2 * mw, frame.shape[0]))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return frame, None
    mr = m.copy()
    mr[:, 2] += mr[:, :2] @ np.array([x0, y0], np.float64)
    mr = mr.astype(np.float32)
    size = (x1 - x0, y1 - y0)
    flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
    rep_b = cv2.warpAffine(rep, mr, size, flags=flags, borderMode=cv2.BORDER_CONSTANT)
    a_b = cv2.warpAffine(a, mr, size, flags=flags, borderMode=cv2.BORDER_CONSTANT, borderValue=0)[..., None]
    out = frame.copy()
    roi = out[y0:y1, x0:x1].astype(np.float32)
    out[y0:y1, x0:x1] = np.clip(roi * (1 - a_b) + rep_b * a_b, 0, 255).astype(np.uint8)
    return out, (x0, y0, a_b[..., 0])


class Capture:
    def __init__(self, cap, mp, lm, cfg):
        self.cap, self.mp, self.lm, self.cfg = cap, mp, lm, cfg
        self.cond = threading.Condition()
        self.item = None
        self.seq = 0
        self.last_ts = 0
        self.running = True
        self.paused = False
        self.idle = threading.Event()
        threading.Thread(target=self.run, daemon=True).start()

    def run(self):
        while self.running:
            if self.paused:
                self.idle.set()
                time.sleep(0.01)
                continue
            self.idle.clear()
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            ts = max(int(time.monotonic() * 1000), self.last_ts + 1)
            self.last_ts = ts
            pts = detect(self.mp, self.lm, frame, ts) if self.cfg.enabled else None
            with self.cond:
                self.item = (frame, pts)
                self.seq += 1
                self.cond.notify_all()
        self.idle.set()

    def get(self, last):
        with self.cond:
            self.cond.wait_for(lambda: self.seq != last, timeout=0.5)
            return self.seq, self.item

    def pause(self):
        self.paused = True
        self.idle.wait(2.0)

    def resume(self, cap):
        self.cap = cap
        self.paused = False

    def stop(self):
        self.running = False
        self.idle.wait(2.0)


class FrameHub:
    def __init__(self):
        self.cond = threading.Condition()
        self.jpg = None
        self.seq = 0

    def put(self, jpg):
        with self.cond:
            self.jpg = jpg
            self.seq += 1
            self.cond.notify_all()

    def get(self, last):
        with self.cond:
            self.cond.wait_for(lambda: self.seq != last, timeout=1.0)
            return self.seq, self.jpg


PAGE = b"<!doctype html><html><body style='margin:0;background:#000;overflow:hidden'><img src='/stream' style='width:100vw;height:100vh;object-fit:contain;display:block'></body></html>"


def make_handler(hub):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            if self.path.startswith("/stream"):
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                last = -1
                try:
                    while True:
                        last, jpg = hub.get(last)
                        if jpg is None:
                            continue
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                    pass
            else:
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(PAGE)))
                self.end_headers()
                self.wfile.write(PAGE)

    return Handler


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading face model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def make_landmarker(video):
    import mediapipe as mp
    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision as mpv
    opts = mpv.FaceLandmarkerOptions(
        base_options=mpp.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mpv.RunningMode.VIDEO if video else mpv.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp, mpv.FaceLandmarker.create_from_options(opts)


def detect(mp, lm, bgr, ts=None):
    h, w = bgr.shape[:2]
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    res = lm.detect(img) if ts is None else lm.detect_for_video(img, ts)
    if not res.face_landmarks:
        return None
    return np.array([[p.x * w, p.y * h, p.z * w] for p in res.face_landmarks[0]], np.float64)


def open_camera(index, w, h, fps):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) if sys.platform == "win32" else cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


CAMERA_FILE = os.path.join(HERE, "last_camera.txt")


def list_cameras():
    if sys.platform == "win32":
        try:
            from pygrabber.dshow_graph import FilterGraph
            return list(enumerate(FilterGraph().get_input_devices()))
        except Exception:
            pass
    found = []
    for i in range(10):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW) if sys.platform == "win32" else cv2.VideoCapture(i)
        if cap.isOpened() and cap.read()[0]:
            found.append((i, f"Camera {i}"))
        cap.release()
    return found


def is_virtual(name):
    n = name.lower()
    return "obs virtual" in n or "obs-camera" in n


def load_saved_camera():
    try:
        with open(CAMERA_FILE, encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return None


def save_camera(name):
    try:
        with open(CAMERA_FILE, "w", encoding="utf-8") as f:
            f.write(name)
    except OSError:
        pass


def print_cameras(cams):
    if not cams:
        print("No cameras found.")
    for n, (idx, name) in enumerate(cams, 1):
        print(f"  {n}) {name}{'   (OBS output)' if is_virtual(name) else ''}   [index {idx}]")


def choose_camera(args, cams):
    if args.camera is not None:
        return args.camera, next((n for i, n in cams if i == args.camera), f"Camera {args.camera}")
    real = cams
    if args.camera_name:
        hit = [c for c in cams if args.camera_name.lower() in c[1].lower()]
        if hit:
            save_camera(hit[0][1])
            return hit[0]
        print(f"No camera matching '{args.camera_name}'.")
    if not real:
        return 0, "Camera 0"
    saved = load_saved_camera()
    if saved and not args.pick:
        hit = [c for c in real if c[1] == saved]
        if hit:
            return hit[0]
    if len(real) == 1 and not args.pick:
        return real[0]
    print("Select a webcam:")
    print_cameras(real)
    while True:
        try:
            raw = input(f"Enter 1-{len(real)} [1]: ").strip() or "1"
        except EOFError:
            raw = "1"
        if raw.isdigit() and 1 <= int(raw) <= len(real):
            choice = real[int(raw) - 1]
            save_camera(choice[1])
            return choice


class GlobalHotkey:
    MODS = {"ctrl": 0x2, "control": 0x2, "alt": 0x1, "shift": 0x4, "win": 0x8}

    def __init__(self, spec, callback):
        self.mods, self.vk = self.parse(spec)
        self.callback = callback
        self.thread_id = None
        self.ok = False
        self.ready = threading.Event()
        threading.Thread(target=self.run, daemon=True).start()
        self.ready.wait(2.0)

    @classmethod
    def parse(cls, spec):
        parts = [p.strip().lower() for p in spec.split("+") if p.strip()]
        mods = 0
        for p in parts[:-1]:
            mods |= cls.MODS[p]
        key = parts[-1]
        if len(key) == 1 and key.isalnum():
            return mods, ord(key.upper())
        if key[0] == "f" and key[1:].isdigit() and 1 <= int(key[1:]) <= 24:
            return mods, 0x6F + int(key[1:])
        raise ValueError(spec)

    def run(self):
        import ctypes
        from ctypes import wintypes
        user32 = ctypes.windll.user32
        self.thread_id = ctypes.windll.kernel32.GetCurrentThreadId()
        self.ok = bool(user32.RegisterHotKey(None, 1, self.mods | 0x4000, self.vk))
        self.ready.set()
        if not self.ok:
            return
        msg = wintypes.MSG()
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            if msg.message == 0x0312:
                self.callback()
        user32.UnregisterHotKey(None, 1)

    def stop(self):
        if self.ok:
            import ctypes
            ctypes.windll.user32.PostThreadMessageW(self.thread_id, 0x0012, 0, 0)


def start_hotkey(spec, cfg):
    if spec.lower() == "none" or sys.platform != "win32":
        return None

    def toggle():
        cfg.enabled = not cfg.enabled
        print(f"Filter {'ON' if cfg.enabled else 'OFF'}")

    try:
        hotkey = GlobalHotkey(spec, toggle)
    except (KeyError, ValueError, IndexError):
        print(f"Unrecognized --hotkey '{spec}'. Use modifiers + one letter, digit or F-key, e.g. ctrl+alt+f or f9.")
        return None
    if not hotkey.ok:
        print(f"Could not register the hotkey {spec}; another app may be using it. Pick another with --hotkey, e.g. ctrl+alt+t.")
        return None
    cfg.hotkey = spec
    print(f"Press {spec} in any app to turn the filter off/on.")
    return hotkey


def draw_hud(img, cfg, fps, has_ref, recording):
    toggle = f"[f / {cfg.hotkey}]" if cfg.hotkey else "[f]"
    lines = [
        f"filter {'ON' if cfg.enabled else 'OFF'} {toggle}   debug [d]   hud [h]   record {'ON' if recording else 'off'} [r]   quit [q]",
        f"max gap {cfg.max_gap:.2f} [ [ / ] ]   min gap {cfg.min_gap:.3f} [ - / = ]   extent {cfg.extent:.2f} [ , / . ]",
        f"upper {'on' if cfg.upper else 'off'} [u]   lower {'on' if cfg.lower else 'off'} [l]   source {cfg.source} [p]{'' if has_ref else ' (no --ref)'}",
        f"method: {cfg.method}   auto-calibrate {'ON' if cfg.auto else 'off'} [a]   fill: {cfg.fill}{'' if cfg.ai is not None else ' (no AI model)'} [m]   camera: {cfg.camera_name} [c = next]",
        f"{fps:.1f} fps   settings.json saved on quit / when auto-calibrate stops   reset learned gaps [x]",
    ]
    s = img.shape[1] / 1280
    for i, t in enumerate(lines):
        y = int((24 + i * 24) * s)
        cv2.putText(img, t, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * s, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, t, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * s, (255, 255, 255), 1, cv2.LINE_AA)


def draw_label(img, text, color):
    s = img.shape[1] / 1280
    yb = img.shape[0] - int(16 * s)
    cv2.putText(img, text, (12, yb), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * s, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (12, yb), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * s, color, 1, cv2.LINE_AA)


class Recorder:
    def __init__(self, size, fps):
        os.makedirs(RECORD_DIR, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        four = cv2.VideoWriter_fourcc(*"mp4v")
        self.raw_path = os.path.join(RECORD_DIR, f"{stamp}_raw.mp4")
        self.out_path = os.path.join(RECORD_DIR, f"{stamp}_filtered.mp4")
        self.raw = cv2.VideoWriter(self.raw_path, four, fps, size)
        self.out = cv2.VideoWriter(self.out_path, four, fps, size)

    def write(self, raw, out):
        self.raw.write(raw)
        self.out.write(out)

    def close(self):
        self.raw.release()
        self.out.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=None)
    ap.add_argument("--camera-name", dest="camera_name", type=str, default=None)
    ap.add_argument("--pick", action="store_true")
    ap.add_argument("--list-cameras", dest="list_cameras", action="store_true")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--hotkey", type=str, default="ctrl+alt+f")
    ap.add_argument("--ref", type=str, default=None)
    ap.add_argument("--dark", type=float, default=None)
    ap.add_argument("--extent", type=float, default=None)
    ap.add_argument("--max-gap", dest="max_gap", type=float, default=None)
    ap.add_argument("--min-gap", dest="min_gap", type=float, default=None)
    ap.add_argument("--mirror-extent", dest="mirror_extent", type=float, default=None)
    ap.add_argument("--min-open", dest="min_open", type=float, default=0.05)
    ap.add_argument("--no-upper", action="store_true")
    ap.add_argument("--no-lower", action="store_true")
    ap.add_argument("--auto", action="store_true")
    ap.add_argument("--method", choices=["anchor", "edge", "legacy"], default="anchor")
    ap.add_argument("--fill", choices=["auto", "ai", "mirror"], default="auto")
    ap.add_argument("--window-width", dest="window_width", type=int, default=852)
    ap.add_argument("--window-height", dest="window_height", type=int, default=480)
    ap.add_argument("--image", type=str, default=None)
    ap.add_argument("--video", type=str, default=None)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    if args.list_cameras:
        print_cameras(list_cameras())
        return

    saved, priors = load_settings()
    for key, default in DEFAULTS.items():
        if getattr(args, key) is None:
            setattr(args, key, saved.get(key, default))

    ensure_model()
    cfg = Config(args)
    if args.fill != "mirror":
        cfg.ai = load_ai()
        if cfg.ai is not None:
            cfg.fill = "ai"
            print(f"AI fill loaded ({cfg.ai.provider}). Press [m] to switch between AI and mirror fill.")
        elif args.fill == "ai":
            print("No teeth_ai.onnx found; run train_ai.bat first. Using mirror fill.")

    proc = (build_procedural(True), build_procedural(False))
    ref = None
    if args.ref:
        mp, lm_img = make_landmarker(False)
        rimg = cv2.imread(args.ref)
        rpts = detect(mp, lm_img, rimg) if rimg is not None else None
        if rpts is None:
            print("Could not find a face in the reference image; using procedural teeth as the fallback.")
        else:
            r = build_reference(rimg, rpts[:, :2])
            ref = (r[0] if r[0] is not None else proc[0], r[1] if r[1] is not None else proc[1])
        lm_img.close()

    def atlases():
        return ref if (ref is not None and cfg.source != "procedural") else proc

    if args.image:
        mp, lm_img = make_landmarker(False)
        img = cv2.imread(args.image)
        pts = detect(mp, lm_img, img)
        lm_img.close()
        if pts is None:
            print("No face found.")
            return
        st = State(priors)
        anchor = make_anchor(cfg, st.prior)
        out = img
        for _ in range(4):
            out, _ = filter_frame(img, pts, st, cfg, atlases(), anchor)
        path = args.out or "filtered.png"
        cv2.imwrite(path, out)
        print("Wrote", path)
        return

    if args.video:
        mp, lm_v = make_landmarker(True)
        cap = cv2.VideoCapture(args.video)
        fps_v = cap.get(cv2.CAP_PROP_FPS) or 30
        path = args.out or os.path.splitext(args.video)[0] + "_filtered.mp4"
        writer = None
        st = State(priors)
        anchor = make_anchor(cfg, st.prior)
        n = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if writer is None:
                writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps_v, (frame.shape[1], frame.shape[0]))
            pts = detect(mp, lm_v, frame, int(n * 1000 / fps_v) + 1)
            out = frame
            if pts is not None:
                out, _ = filter_frame(frame, pts, st, cfg, atlases(), anchor)
            else:
                st.pts = None
            writer.write(out)
            n += 1
        cap.release()
        lm_v.close()
        if writer is not None:
            writer.release()
        print(f"Wrote {path} ({n} frames)")
        return

    mp, lm = make_landmarker(True)
    cams = list_cameras()
    cam_idx, cfg.camera_name = choose_camera(args, cams)
    print(f"Using camera: {cfg.camera_name}")
    if is_virtual(cfg.camera_name):
        print("Note: using OBS Virtual Camera as input. Point OBS's virtual camera at a scene or source that does NOT contain the Teeth Filter browser source, or the output will loop back into itself.")
    cap = open_camera(cam_idx, args.width, args.height, args.fps)
    if not cap.isOpened():
        print("Could not open camera. Close OBS/other apps that are using it, or run: run.bat --pick")
        return
    hub = FrameHub()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(hub))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"OBS Browser Source URL: http://127.0.0.1:{args.port}/")
    st = State(priors)
    anchor = make_anchor(cfg, st.prior)
    win = "Teeth Filter"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, args.window_width, args.window_height)
    fps = 0.0
    t_prev = time.perf_counter()
    enc = [int(cv2.IMWRITE_JPEG_QUALITY), 92]
    rec = None
    noface = 0
    grab = Capture(cap, mp, lm, cfg)
    hotkey = start_hotkey(args.hotkey, cfg)
    was_enabled = cfg.enabled
    seq = -1
    while True:
        seq, item = grab.get(seq)
        if item is None:
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
            continue
        frame, pts = item
        out, dbg = frame, None
        if cfg.enabled and not was_enabled and anchor is not None:
            anchor.clear_holds()
        was_enabled = cfg.enabled
        if cfg.enabled:
            noface = 0 if pts is not None else noface + 1
            if pts is not None:
                out, dbg = filter_frame(frame, pts, st, cfg, atlases(), anchor)
            else:
                st.pts = None
                st.alpha = None
        else:
            noface = 0
        ok, jpg = cv2.imencode(".jpg", out, enc)
        if ok:
            hub.put(jpg.tobytes())
        if rec is not None:
            rec.write(frame, out)
        view = out
        if cfg.debug and dbg is not None:
            view = out.copy()
            x0, y0, a = dbg
            h, w = a.shape
            roi = view[y0:y0 + h, x0:x0 + w].astype(np.float32)
            roi[..., 2] = np.clip(roi[..., 2] + a * 200, 0, 255)
            view[y0:y0 + h, x0:x0 + w] = roi.astype(np.uint8)
            cv2.rectangle(view, (x0, y0), (x0 + w, y0 + h), (0, 255, 255), 1)
        if cfg.hud or cfg.auto or rec is not None or noface > 15 or not cfg.enabled:
            view = view.copy() if view is out else view
        if cfg.hud:
            draw_hud(view, cfg, fps, ref is not None, rec is not None)
        if not cfg.enabled:
            keys = f"[f] or {cfg.hotkey}" if cfg.hotkey else "[f]"
            draw_label(view, f"FILTER OFF - {keys} to turn it back on" + ("   REC [r] to stop" if rec is not None else ""), (0, 200, 255))
        elif noface > 15:
            draw_label(view, "NO FACE DETECTED - keep your whole face (eyes to chin) in frame; the filter is paused", (0, 0, 255))
        elif cfg.auto:
            draw_label(view, f"AUTO-CALIBRATING  extent {cfg.extent:.2f}  ({st.calib_frames} samples)  [a] to stop", (0, 220, 255))
        elif rec is not None:
            draw_label(view, "REC  [r] to stop", (0, 0, 255))
        cv2.imshow(win, view)
        t = time.perf_counter()
        fps = 0.9 * fps + 0.1 / max(t - t_prev, 1e-6)
        t_prev = t
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break
        if k == ord("f"):
            cfg.enabled = not cfg.enabled
        elif k == ord("d"):
            cfg.debug = not cfg.debug
        elif k == ord("r"):
            if rec is None:
                rec = Recorder((frame.shape[1], frame.shape[0]), args.fps)
                print(f"Recording to {rec.raw_path} and {rec.out_path}")
            else:
                rec.close()
                print("Recording saved.")
                rec = None
        elif k == ord("a"):
            cfg.auto = not cfg.auto
            if cfg.auto:
                st.calib_frames = 0
                print("Auto-calibration ON: talk and smile naturally, facing the camera. Press 'a' to stop.")
            else:
                data = save_settings(cfg, st, anchor)
                print(f"Auto-calibration OFF. Saved to settings.json: {data}")
        elif k == ord("x"):
            st.prior = {}
            st.prior_ready = False
            print("Learned gap positions cleared.")
        elif k == ord("c"):
            real = cams
            if len(real) > 1:
                pos = next((n for n, c in enumerate(real) if c[0] == cam_idx), -1)
                grab.pause()
                for step in range(1, len(real) + 1):
                    nxt = real[(pos + step) % len(real)]
                    cap.release()
                    cap = open_camera(nxt[0], args.width, args.height, args.fps)
                    if cap.isOpened():
                        cam_idx, cfg.camera_name = nxt
                        save_camera(cfg.camera_name)
                        st = State(st.prior)
                        if anchor is not None:
                            anchor.reset()
                        break
                grab.resume(cap)
                print(f"Using camera: {cfg.camera_name}")
        elif k == ord("m") and cfg.ai is not None:
            cfg.fill = "mirror" if cfg.fill == "ai" else "ai"
            print("Fill:", cfg.fill)
        elif k == ord("h"):
            cfg.hud = not cfg.hud
        elif k == ord("u"):
            cfg.upper = not cfg.upper
        elif k == ord("l"):
            cfg.lower = not cfg.lower
        elif k == ord("p"):
            options = SOURCES if ref is not None else ["self", "procedural"]
            cfg.source = options[(options.index(cfg.source) + 1) % len(options)] if cfg.source in options else "self"
        elif k == ord("["):
            cfg.max_gap = max(0.08, cfg.max_gap - 0.02)
        elif k == ord("]"):
            cfg.max_gap = min(0.5, cfg.max_gap + 0.02)
        elif k == ord("-"):
            cfg.min_gap = max(0.01, cfg.min_gap - 0.005)
        elif k == ord("="):
            cfg.min_gap = min(0.15, cfg.min_gap + 0.005)
        elif k == ord(","):
            cfg.extent = max(0.3, cfg.extent - 0.05)
        elif k == ord("."):
            cfg.extent = min(1.0, cfg.extent + 0.05)
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
    if rec is not None:
        rec.close()
    save_settings(cfg, st, anchor)
    if hotkey is not None:
        hotkey.stop()
    grab.stop()
    grab.cap.release()
    lm.close()
    server.shutdown()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
