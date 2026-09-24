import os

import cv2
import numpy as np

import common as C
from render_v10 import OneEuro, highpass, lip_curve, UPPER_INNER, LOWER_INNER

INNER_POLY = UPPER_INNER + LOWER_INNER[::-1][1:-1]
from edge_track import EdgeTracker

CW, CH = C.CW, C.CH
GY, GX = np.mgrid[0:CH, 0:CW].astype(np.float32)


def shift_img(img, d, border=cv2.BORDER_REPLICATE, value=0):
    m = np.float32([[1, 0, d[0]], [0, 1, d[1]]])
    return cv2.warpAffine(img, m, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=border, borderValue=value)


class Model:
    def __init__(self, path=None):
        z = np.load(path or os.path.join(C.CACHE, "model_v10c.npz"))
        for k in z.files:
            setattr(self, k, z[k])
        self.x_mid = float(self.x_mid)
        self.tpl_box = tuple(int(v) for v in self.tpl_box)
        self.ref_box = tuple(int(v) for v in self.ref_box)
        self.slot = self.slot.astype(np.float32)
        b = self.bottom.astype(np.float32).copy()
        xs = np.arange(len(b))
        mir = np.interp(2 * self.x_mid - xs, xs, b, left=0, right=0)
        full = np.where(xs < self.x_mid, mir, b)
        good = full > 0
        self.bottom_full = np.interp(xs, xs[good], full[good]).astype(np.float32) if good.any() else np.full(len(b), 130, np.float32)


class Renderer:
    def __init__(self, model, df=None, live=True, atlas=True, extend=True, lower=True):
        self.m = model
        self.lower = lower
        self.alpha_l_prev = None
        self.extend = extend
        self.df = df or C.dental()
        self.use_live = live
        self.use_atlas = atlas
        self.filt = OneEuro(1.0, 0.02)
        self.et = EdgeTracker(model.x_mid + model.edge_offset, float(model.bottom[int(model.x_mid) + 18]))
        self.delta = None
        self.delta_med = None
        self.miss = 99
        self.gain = None
        self.alpha_prev = None
        self.thr = None
        self.score = 0.0
        self.gate = 4.0
        self.conf_hi = 0.12
        self.conf_lo = 0.06
        self.locks = 0
        self.last_hidden = None
        self.band = 12
        self.outliers = []

    def anchor(self, a0, crop0, mouth0, hide=False):
        meas = self.et.measure(crop0, mouth0)
        base = np.array([self.et.x0, self.et.b0])
        ref = self.delta if (self.delta is not None and self.miss < 30) else self.et.off
        good = False
        if meas is not None and meas["conf"] >= self.conf_lo:
            d_m = np.array([meas["x"] - base[0], (meas["y"] - base[1]) if meas["y"] is not None else ref[1]])
            if meas["y"] is not None and abs(d_m[1] - ref[1]) > 8:
                d_m[1] = ref[1]
            err = abs(d_m[0] - ref[0])
            gate = min(self.gate + 0.25 * self.miss, 12.0)
            if self.et.n < 15:
                good = meas["conf"] >= self.conf_hi
            elif meas["conf"] >= self.conf_hi and err <= gate:
                good = True
            elif err <= 2.5:
                good = True
            elif meas["conf"] >= self.conf_hi:
                self.outliers.append(d_m[0])
                if len(self.outliers) >= 3 and np.ptp(self.outliers[-3:]) < 4:
                    good = True
            if good:
                self.outliers = []
            self.last_meas = d_m.copy()
        self.last_hidden = None
        if good and hide:
            self.last_hidden = d_m.copy()
            good = False
        if good:
            self.et.update(meas)
            loc = base + d_m
            self.miss = 0
            self.locks += 1
        else:
            loc = base + (self.delta if self.delta is not None else self.et.off)
            self.miss += 1
            if self.miss > 8:
                loc = loc + 0.05 * ((base + self.et.off) - loc)
        p = a0[:, :2] @ loc + a0[:, 2]
        if not good and self.filt.x is not None:
            self.filt.x = p.copy()
            self.filt.dx = np.zeros(2)
        pf = self.filt(p)
        inv = np.linalg.inv(np.vstack([a0, [0, 0, 1]]))
        loc_f = inv[:2, :2] @ pf + inv[:2, 2]
        self.delta = loc_f - base
        self.score = meas["conf"] if (meas is not None and good) else 0.0

    def clean_mouth(self, mouth0, cp0):
        poly = np.round(cp0[INNER_POLY, :2]).astype(np.int32)
        band = np.zeros((CH, CW), np.uint8)
        cv2.fillPoly(band, [poly], 1)
        band = cv2.dilate(band, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * self.band + 1, 2 * self.band + 1)))
        band = cv2.GaussianBlur(band.astype(np.float32), (0, 0), 2.0)
        m = mouth0 * band
        lab_n, labels, stats, _ = cv2.connectedComponentsWithStats((m > 0.5).astype(np.uint8), connectivity=8)
        if lab_n > 2:
            keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            big = cv2.dilate((labels == keep).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(np.float32)
            m = m * big
        return m

    def lower_fill(self, crop, lab, mouth, span=80):
        xm = self.m.x_mid
        L = cv2.GaussianBlur(lab[..., 0], (0, 0), 0.7)
        inm = mouth > 0.5
        ref = self.m.donor_mask > 0.5
        vis_ref = ref & (mouth > 0.7)
        if vis_ref.sum() < 20:
            return None
        lt = float(np.percentile(L[vis_ref], 70))
        thr = 0.72 * lt
        rows = np.arange(CH)
        has = inm.any(0)
        lb = np.where(has, CH - 1 - np.argmax(inm[::-1], 0), -1)
        ub = self.m.bottom_full
        top_l = np.full(CW, np.nan, np.float32)
        bot_l = np.full(CW, np.nan, np.float32)
        for xd in range(int(xm - span), int(xm)):
            b = lb[xd]
            if b < 0:
                continue
            y = b - 1
            while y > b - 6 and L[y, xd] < thr:
                y -= 1
            bottom_tooth = y
            n = 0
            while y > ub[xd] + 1 and inm[y, xd] and L[y, xd] >= thr:
                y -= 1
                n += 1
            if n >= 4:
                top_l[xd] = y + 1
                bot_l[xd] = bottom_tooth
        ok = ~np.isnan(top_l)
        if ok.sum() < 8:
            return None
        cols = np.flatnonzero(ok)
        k = 5
        tp = np.pad(top_l[cols], k // 2, mode="edge")
        top_l[cols] = np.median(np.lib.stride_tricks.sliding_window_view(tp, k), 1)
        xs_all = np.arange(CW)
        xd_f = 2 * xm - xs_all
        xdi = np.clip(np.round(xd_f).astype(int), 0, CW - 1)
        t_s = np.where((xs_all > xm) & (xs_all < xm + span), top_l[xdi], np.nan)
        b_d = np.where(np.isnan(bot_l[xdi]), np.nan, bot_l[xdi])
        valid = ~np.isnan(t_s)
        t_s = np.where(valid, t_s, CH).astype(np.float32)
        low = np.maximum(t_s - 1.0, ub + 1.0)
        alpha = (GY >= low[None, :]).astype(np.float32) * np.clip((mouth - 0.3) / 0.4, 0, 1) * valid[None, :]
        mapy = np.minimum(GY, np.nan_to_num(b_d, nan=CH)[None, :]).astype(np.float32)
        tex = cv2.remap(crop, xd_f[None, :].repeat(CH, 0).astype(np.float32), mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
        return tex, cv2.GaussianBlur(alpha, (0, 0), 0.6)

    def collect(self, crop, lab, tex, alpha, mouth):
        vis_slot = (self.m.slot > 0.5) & (mouth > 0.6)
        if not hasattr(self, "stats"):
            self.stats = []
            self.prev_comp = None
        comp = crop.astype(np.float32) * (1 - alpha[..., None]) + tex * alpha[..., None]
        L = cv2.cvtColor(np.clip(comp, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
        s = {"vis": int(vis_slot.sum()), "miss": self.miss}
        donor = (self.m.donor_mask > 0.5) & (mouth > 0.6)
        if vis_slot.sum() > 30 and donor.sum() > 30:
            tooth_l = float(np.percentile(L[donor], 60))
            s["leak_raw"] = float((lab[..., 0][vis_slot] < 0.6 * tooth_l).mean())
            s["leak"] = float((L[vis_slot] < 0.6 * tooth_l).mean())
            s["cover"] = float((alpha[vis_slot] > 0.5).mean())
        if self.prev_comp is not None:
            both = (alpha > 0.5) & (self.prev_alpha_m > 0.5)
            don2 = donor & self.prev_donor
            if both.sum() > 30 and don2.sum() > 30:
                s["flick_fill"] = float(np.abs(L - self.prev_comp)[both].mean())
                s["flick_real"] = float(np.abs(L - self.prev_comp)[don2].mean())
        self.prev_comp = L
        self.prev_alpha_m = alpha
        self.prev_donor = donor
        self.stats.append(s)

    def summary(self):
        st = getattr(self, "stats", [])
        f = lambda k: [s[k] for s in st if k in s]
        out = {"frames": len(st), "vis_frames": len(f("leak"))}
        for k in ("leak_raw", "leak", "cover", "flick_fill", "flick_real"):
            v = f(k)
            out[k] = float(np.mean(v)) if v else float("nan")
        out["flick_ratio"] = out["flick_fill"] / out["flick_real"] if out["flick_real"] == out["flick_real"] and out["flick_real"] > 0 else float("nan")
        return out

    def process(self, frame, pts, mouth0, debug=False, hide=False):
        df = self.df
        pose = df.pose(pts)
        a0 = df.affine(pose)
        flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
        crop0 = cv2.warpAffine(frame, a0.astype(np.float32), (CW, CH), flags=flags, borderMode=cv2.BORDER_REPLICATE)
        mouth0 = self.clean_mouth(mouth0, df.to_crop(pose, pts))
        self.anchor(a0, crop0, mouth0, hide)
        score = self.score
        a = a0.copy()
        a[:, 2] += a0[:, :2] @ self.delta
        crop = cv2.warpAffine(frame, a.astype(np.float32), (CW, CH), flags=flags, borderMode=cv2.BORDER_REPLICATE)
        mouth = shift_img(mouth0, self.delta, cv2.BORDER_CONSTANT, 0)
        xm = self.m.x_mid
        body = self.m.slot
        alpha = body * np.clip((mouth - 0.3) / 0.4, 0, 1)
        xd = (2 * xm - GX).astype(np.float32)
        mouth_d = cv2.remap(mouth, xd, GY, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        mapy = GY
        if self.extend:
            body_top = np.argmax(body > 0.5, 0)
            vis_d = (mouth_d > 0.65) & (body > 0.5)
            has = vis_d.any(0)
            top_d = np.where(has, np.argmax(vis_d, 0), CH).astype(np.float32)
            top_d = np.maximum(top_d, body_top)
            mapy = np.maximum(GY, (top_d + 1.0)[None, :]).astype(np.float32)
            mouth_d = np.where(GY < top_d[None, :] + 1, np.where(has[None, :], 1.0, mouth_d), mouth_d).astype(np.float32)
        live = cv2.remap(crop, xd, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
        tex = live
        if self.use_atlas:
            don = (self.m.donor_mask > 0.5) & (mouth > 0.8) & (self.m.atlas_bgr.sum(2) > 0)
            if don.sum() > 60:
                g = np.array([np.median(crop[..., c][don].astype(np.float32) / np.maximum(self.m.atlas_bgr[..., c][don], 1)) for c in range(3)])
                self.gain = g if self.gain is None else 0.85 * self.gain + 0.15 * g
            gain = self.gain if self.gain is not None else np.ones(3)
            at = self.m.atlas_slot * gain[None, None, :]
            if self.use_live:
                w = np.clip((mouth_d - 0.5) / 0.4, 0, 1)
                w = cv2.erode(w, np.ones((3, 1), np.uint8))[..., None]
                tex = w * live + (1 - w) * at
            else:
                tex = at
        alpha = cv2.GaussianBlur(alpha.astype(np.float32), (0, 0), 0.6)
        cp = df.to_crop(pose, pts) - self.delta
        opening = float(np.median((lip_curve(cp, LOWER_INNER) - lip_curve(cp, UPPER_INNER))[int(xm) - 40:int(xm) + 40]))
        alpha *= float(np.clip((opening - 2) / 5, 0, 1)) * float(np.clip(self.locks / 3.0, 0, 1))
        if self.alpha_prev is not None:
            alpha = np.maximum(alpha, 0.35 * self.alpha_prev * (mouth > 0.5))
        self.alpha_prev = alpha
        self.collect(crop, lab, tex, alpha, mouth)
        if self.lower:
            lf = self.lower_fill(crop, lab, mouth)
            if lf is not None:
                tex_l, alpha_l = lf
                alpha_l *= float(np.clip((opening - 2) / 5, 0, 1))
                if self.alpha_l_prev is not None:
                    alpha_l = 0.6 * alpha_l + 0.4 * self.alpha_l_prev * (mouth > 0.5)
                self.alpha_l_prev = alpha_l
                use_l = alpha_l > alpha
                tex = np.where(use_l[..., None], tex_l, tex)
                alpha = np.maximum(alpha, alpha_l)
            else:
                self.alpha_l_prev = None
        dbg = None
        if debug:
            comp = crop.astype(np.float32) * (1 - alpha[..., None]) + tex * alpha[..., None]
            dbg = {"crop": crop, "comp": np.clip(comp, 0, 255).astype(np.uint8), "alpha": alpha, "mouth": mouth, "score": score, "delta": self.delta.copy()}
        if alpha.max() < 0.01:
            return frame, dbg
        return paste(frame, pts, a, tex, alpha), dbg


def paste(frame, pts, a, tex, alpha):
    m3 = np.vstack([a, [0, 0, 1]])
    poly = pts[:, :2][UPPER_INNER + LOWER_INNER]
    mw = np.linalg.norm(pts[308, :2] - pts[78, :2])
    bx0 = int(max(poly[:, 0].min() - 0.3 * mw, 0)); by0 = int(max(poly[:, 1].min() - 0.3 * mw, 0))
    bx1 = int(min(poly[:, 0].max() + 0.3 * mw, frame.shape[1])); by1 = int(min(poly[:, 1].max() + 0.3 * mw, frame.shape[0]))
    if bx1 - bx0 < 4 or by1 - by0 < 4:
        return frame
    sh = np.array([[1, 0, bx0], [0, 1, by0], [0, 0, 1]], np.float64)
    fwd = (np.linalg.inv(m3) @ sh)[:2].astype(np.float32)
    size = (bx1 - bx0, by1 - by0)
    flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
    rep_b = cv2.warpAffine(tex.astype(np.float32), fwd, size, flags=flags, borderMode=cv2.BORDER_REPLICATE)
    a_b = cv2.warpAffine(alpha, fwd, size, flags=flags, borderMode=cv2.BORDER_CONSTANT, borderValue=0)[..., None]
    out = frame.copy()
    roi = out[by0:by1, bx0:bx1].astype(np.float32)
    out[by0:by1, bx0:bx1] = np.clip(roi * (1 - a_b) + rep_b * a_b, 0, 255).astype(np.uint8)
    return out
