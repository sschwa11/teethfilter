import os
import sys

import cv2
import numpy as np

import common as C
import rigid as R

CW, CH = C.CW, C.CH
GY, GX = np.mgrid[0:CH, 0:CW].astype(np.float32)
UPPER_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
LOWER_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]


def lip_curve(cp, ids):
    p = cp[ids]
    o = np.argsort(p[:, 0])
    return np.interp(np.arange(CW), p[o, 0], p[o, 1]).astype(np.float32)


def highpass(gray, s=3.0):
    return gray - cv2.GaussianBlur(gray, (0, 0), s)


class OneEuro:
    def __init__(self, min_cutoff=0.4, beta=0.02, dcutoff=1.0, fps=30.0):
        self.mc, self.beta, self.dc, self.fps = min_cutoff, beta, dcutoff, fps
        self.x = None
        self.dx = None

    @staticmethod
    def alpha(cutoff, fps):
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau * fps)

    def __call__(self, x):
        x = np.asarray(x, np.float64)
        if self.x is None:
            self.x, self.dx = x.copy(), np.zeros_like(x)
            return self.x
        dx = (x - self.x) * self.fps
        a_d = self.alpha(self.dc, self.fps)
        self.dx = a_d * dx + (1 - a_d) * self.dx
        cutoff = self.mc + self.beta * np.abs(self.dx)
        a = self.alpha(cutoff, self.fps)
        self.x = a * x + (1 - a) * self.x
        return self.x


class Model:
    def __init__(self, path=None):
        z = np.load(path or os.path.join(C.CACHE, "model_v10.npz"))
        self.tpl = z["tpl"].astype(np.float32)
        self.tpl_box = tuple(int(v) for v in z["tpl_box"])
        self.x_mid = float(z["x_mid"])
        self.bottom = z["bottom"].astype(np.float32)
        self.top = z["top"].astype(np.float32)
        self.slot = z["slot"].astype(bool)
        self.atlas = z["atlas"].astype(np.float32)
        self.atlas_ok = z["atlas_ok"].astype(bool)
        self.ref_box = tuple(int(v) for v in z["ref_box"])


class Renderer:
    def __init__(self, model, df=None, live_weight=1.0, mode="runs"):
        self.m = model
        self.mode = mode
        self.ext_prev = None
        self.ext_miss = 0
        self.df = df or C.dental()
        self.delta = None
        self.delta_med = None
        self.filt = OneEuro(0.5, 0.05)
        self.miss = 99
        self.eu_prev = None
        self.alpha_prev = None
        self.ref_color = None
        self.live_weight = live_weight
        self.last_score = 0.0

    def track(self, crop):
        y0, y1, x0, x1 = self.m.tpl_box
        pad = 16
        g = highpass(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32))
        win = g[y0 - pad:y1 + pad, x0 - pad:x1 + pad]
        r = cv2.matchTemplate(win, self.m.tpl, cv2.TM_CCOEFF_NORMED)
        _, mv, _, ml = cv2.minMaxLoc(r)
        px, py = ml
        dx = dy = 0.0
        if 0 < px < r.shape[1] - 1:
            den = r[py, px - 1] - 2 * r[py, px] + r[py, px + 1]
            dx = 0.5 * (r[py, px - 1] - r[py, px + 1]) / den if abs(den) > 1e-6 else 0.0
        if 0 < py < r.shape[0] - 1:
            den = r[py - 1, px] - 2 * r[py, px] + r[py + 1, px]
            dy = 0.5 * (r[py - 1, px] - r[py + 1, px]) / den if abs(den) > 1e-6 else 0.0
        return np.array([px + dx - pad, py + dy - pad]), float(mv)

    def upper_edge(self, lab, yu):
        L = cv2.GaussianBlur(lab[..., 0], (0, 0), 0.8)
        A = cv2.GaussianBlur(lab[..., 1], (0, 0), 0.8)
        dL = np.zeros_like(L)
        dA = np.zeros_like(A)
        dL[1:-1] = (L[2:] - L[:-2]) / 2
        dA[1:-1] = (A[2:] - A[:-2]) / 2
        score = np.clip(-dL, 0, None) + 2.0 * np.clip(-dA, 0, None)
        edge = yu.copy()
        cols = np.flatnonzero(self.m.slot.any(0) | (np.abs(GX[0] - self.m.x_mid) < 110))
        for x in cols:
            y0 = int(max(yu[x] - 9, 1))
            y1 = int(min(max(yu[x] + 12, y0 + 2), self.m.bottom[x] + 2, CH - 2))
            if y1 <= y0 + 1:
                continue
            s = score[y0:y1, x]
            mx = s.max()
            if mx < 4:
                continue
            peaks = np.flatnonzero((s >= 0.5 * mx) & (s >= np.roll(s, 1)) & (s >= np.roll(s, -1)))
            if peaks.size:
                edge[x] = y0 + peaks[0] + 0.5
        k = 9
        pad = np.pad(edge, k // 2, mode="edge")
        edge = np.median(np.lib.stride_tricks.sliding_window_view(pad, k), 1).astype(np.float32)
        return edge

    def lower_edge(self, lab, yl):
        return yl

    def donor_runs(self, lab):
        L = cv2.GaussianBlur(lab[..., 0], (0, 0), 0.7)
        xm = self.m.x_mid
        y0, y1, x0, x1 = self.m.ref_box
        core = L[y0:y1, x0:x1]
        lt = float(np.percentile(core, 90))
        below = L[int(self.m.bottom[x0]) + 4:int(self.m.bottom[x0]) + 18, x0:x1]
        ld = float(np.percentile(below, 20)) if below.size else lt * 0.5
        thr = max(ld + 0.55 * (lt - ld), lt - 45)
        top = np.full(CW, np.nan, np.float32)
        bot = np.full(CW, np.nan, np.float32)
        for xd in range(int(np.ceil(xm)), CW):
            b = self.m.bottom[xd]
            if b <= 0:
                continue
            r0 = int(max(b - 34, 0))
            r1 = int(min(b + 2, CH))
            t = L[r0:r1, xd] > thr
            t = np.convolve(t.astype(np.float32), np.ones(3), "same") >= 2
            idx = np.flatnonzero(t)
            if idx.size < 3:
                continue
            runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
            run = max(runs, key=lambda q: (q[-1] >= len(t) - 8, len(q)))
            if len(run) < 3:
                continue
            top[xd] = r0 + run[0]
            bot[xd] = r0 + run[-1] + 1
        return top, bot, thr, lt

    def slot_extent(self, lab, top_d, bot_d, yu):
        L = cv2.GaussianBlur(lab[..., 0], (0, 0), 0.8)
        dL = np.zeros_like(L)
        dL[1:-1] = (L[2:] - L[:-2]) / 2
        xm = self.m.x_mid
        top = np.full(CW, np.nan, np.float32)
        bot = np.full(CW, np.nan, np.float32)
        slot_cols = np.flatnonzero(self.m.slot.any(0))
        for xs in slot_cols:
            xd = int(round(2 * xm - xs))
            if xd >= CW or np.isnan(top_d[xd]):
                continue
            prior = top_d[xd] - 1.5
            lo = int(max(prior - 7, 1))
            hi = int(min(prior + 7, CH - 2))
            s = -dL[lo:hi, xs]
            j = int(np.argmax(s))
            t = lo + j + 0.5 if s[j] > 3.0 else prior
            top[xs] = 0.5 * t + 0.5 * prior
            bot[xs] = min(bot_d[xd], self.m.bottom[xs])
        ok = ~np.isnan(top)
        if ok.sum() >= 5:
            xs = np.flatnonzero(ok)
            k = 5
            tp = np.pad(top[xs], k // 2, mode="edge")
            top[xs] = np.median(np.lib.stride_tricks.sliding_window_view(tp, k), 1)
        return top, bot

    def paste(self, frame, pts, a, tex, alpha):
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
        rep_b = cv2.warpAffine(tex, fwd, size, flags=flags, borderMode=cv2.BORDER_REPLICATE)
        a_b = cv2.warpAffine(alpha, fwd, size, flags=flags, borderMode=cv2.BORDER_CONSTANT, borderValue=0)[..., None]
        out = frame.copy()
        roi = out[by0:by1, bx0:bx1].astype(np.float32)
        out[by0:by1, bx0:bx1] = np.clip(roi * (1 - a_b) + rep_b * a_b, 0, 255).astype(np.uint8)
        return out

    def process_runs(self, frame, pts, a, crop, lab, opening, score, debug):
        xm = self.m.x_mid
        top_d, bot_d, thr, lt = self.donor_runs(lab)
        top_s, bot_s = self.slot_extent(lab, top_d, bot_d, None)
        cur = np.stack([top_s, bot_s])
        if self.ext_prev is not None:
            both = ~np.isnan(cur) & ~np.isnan(self.ext_prev)
            cur = np.where(both, 0.5 * cur + 0.5 * self.ext_prev, cur)
            hold = np.isnan(cur) & ~np.isnan(self.ext_prev)
            if self.ext_miss < 3:
                cur = np.where(hold, self.ext_prev, cur)
        valid = ~np.isnan(cur[0])
        self.ext_miss = 0 if valid.any() else self.ext_miss + 1
        self.ext_prev = cur
        top_s = np.where(valid, cur[0], CH)
        bot_s = np.where(valid, cur[1], 0)
        alpha = np.clip(GY - top_s[None, :] + 0.5, 0, 1) * np.clip(bot_s[None, :] - GY + 0.5, 0, 1) * self.m.slot
        xd = (2 * xm - GX).astype(np.float32)
        xdi = np.clip(np.round(2 * xm - GX[0]).astype(int), 0, CW - 1)
        td = np.nan_to_num(top_d[xdi], nan=0.0)
        mapy = np.maximum(GY, td[None, :] + 1).astype(np.float32)
        tex = cv2.remap(crop, xd, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
        alpha = cv2.GaussianBlur(alpha.astype(np.float32), (0, 0), 0.6)
        fade = float(np.clip((opening - 4) / 5, 0, 1))
        alpha *= fade
        if self.alpha_prev is not None:
            alpha = np.where(alpha > self.alpha_prev, alpha, 0.5 * alpha + 0.5 * self.alpha_prev * (alpha > 0.01))
        self.alpha_prev = alpha
        dbg = None
        if debug:
            comp = crop.astype(np.float32) * (1 - alpha[..., None]) + tex * alpha[..., None]
            dbg = {"crop": crop, "comp": np.clip(comp, 0, 255).astype(np.uint8), "alpha": alpha, "eu": np.nan_to_num(top_d + 0, nan=0), "el": np.nan_to_num(bot_d, nan=0),
                   "top_s": top_s, "bot_s": bot_s, "score": score, "delta": self.delta.copy(), "thr": thr}
        if alpha.max() < 0.01:
            return frame, dbg
        return self.paste(frame, pts, a, tex, alpha), dbg

    def process(self, frame, pts, debug=False):
        df = self.df
        pose = df.pose(pts)
        a0 = df.affine(pose)
        crop0 = cv2.warpAffine(frame, a0.astype(np.float32), (CW, CH), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
        cp0 = df.to_crop(pose, pts)
        opening = float(np.median((lip_curve(cp0, LOWER_INNER) - lip_curve(cp0, UPPER_INNER))[int(self.m.x_mid) - 40:int(self.m.x_mid) + 40]))
        d_meas, score = self.track(crop0)
        self.last_score = score
        ok = score > 0.55 and opening > 6
        if ok and self.delta_med is not None and np.abs(d_meas - self.delta_med).max() > 12:
            ok = False
        if ok:
            self.delta_med = d_meas if self.delta_med is None else 0.97 * self.delta_med + 0.03 * d_meas
            self.delta = self.filt(d_meas)
            self.miss = 0
        else:
            self.miss += 1
            if self.delta is None:
                self.delta = np.zeros(2) if self.delta_med is None else self.delta_med.copy()
            elif self.delta_med is not None:
                self.delta = self.delta + 0.03 * (self.delta_med - self.delta)
                self.filt.x = self.delta.copy()
        a = a0.copy()
        a[:, 2] += a0[:, :2] @ self.delta
        crop = cv2.warpAffine(frame, a.astype(np.float32), (CW, CH), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
        cp = cp0 - self.delta
        yu = lip_curve(cp, UPPER_INNER)
        yl = lip_curve(cp, LOWER_INNER)
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
        if self.mode == "runs":
            return self.process_runs(frame, pts, a, crop, lab, opening, score, debug)
        eu = self.upper_edge(lab, yu)
        if self.eu_prev is not None:
            eu = 0.5 * eu + 0.5 * self.eu_prev
        self.eu_prev = eu
        el = self.lower_edge(lab, yl)
        vis = np.clip(GY - eu[None, :] + 0.5, 0, 1) * np.clip(el[None, :] - GY + 0.5, 0, 1)
        slot_bot = self.m.bottom[None, :]
        body = np.clip(slot_bot - GY + 0.5, 0, 1) * self.m.slot
        xm = self.m.x_mid
        donor_x = (2 * xm - GX).astype(np.float32)
        live = cv2.remap(crop, donor_x, GY, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
        donor_vis = cv2.remap(vis.astype(np.float32), donor_x, GY, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        y0, y1, x0, x1 = self.m.ref_box
        rl = lab[y0:y1, x0:x1].reshape(-1, 3)
        rv = vis[y0:y1, x0:x1].ravel() > 0.9
        bright = rl[:, 0] > np.percentile(rl[rv, 0], 40) if rv.sum() > 30 else None
        if bright is not None and (rv & bright).sum() > 30:
            ref = np.median(rl[rv & bright], 0)
            self.ref_color = ref if self.ref_color is None else 0.8 * self.ref_color + 0.2 * ref
        if self.ref_color is None:
            self.ref_color = np.array([190.0, 132.0, 140.0])
        at = self.m.atlas
        atlab = np.dstack([at[..., 0] * self.ref_color[0], at[..., 1] + self.ref_color[1], at[..., 2] + self.ref_color[2]])
        atbgr = cv2.cvtColor(np.clip(atlab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)
        wl = (np.clip((donor_vis - 0.5) * 2, 0, 1) * self.live_weight)[..., None]
        tex = wl * live + (1 - wl) * atbgr
        alpha = body * vis
        alpha = cv2.GaussianBlur(alpha, (0, 0), 0.7)
        fade = float(np.clip((opening - 5) / 6, 0, 1))
        alpha *= fade
        if self.miss > 45:
            alpha *= max(0.0, 1 - (self.miss - 45) / 15)
        comp = crop.astype(np.float32) * (1 - alpha[..., None]) + tex * alpha[..., None]
        if alpha.max() < 0.01:
            return frame, None
        m3 = np.vstack([a, [0, 0, 1]])
        poly = pts[:, :2][UPPER_INNER + LOWER_INNER]
        mw = np.linalg.norm(pts[308, :2] - pts[78, :2])
        bx0 = int(max(poly[:, 0].min() - 0.3 * mw, 0)); by0 = int(max(poly[:, 1].min() - 0.3 * mw, 0))
        bx1 = int(min(poly[:, 0].max() + 0.3 * mw, frame.shape[1])); by1 = int(min(poly[:, 1].max() + 0.3 * mw, frame.shape[0]))
        inv = np.linalg.inv(m3)
        sh = np.array([[1, 0, bx0], [0, 1, by0], [0, 0, 1]], np.float64)
        fwd = (inv @ sh)[:2]
        size = (bx1 - bx0, by1 - by0)
        rep_b = cv2.warpAffine(tex, fwd.astype(np.float32), size, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
        a_b = cv2.warpAffine(alpha, fwd.astype(np.float32), size, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT, borderValue=0)[..., None]
        out = frame.copy()
        roi = out[by0:by1, bx0:bx1].astype(np.float32)
        out[by0:by1, bx0:bx1] = np.clip(roi * (1 - a_b) + rep_b * a_b, 0, 255).astype(np.uint8)
        dbg = None
        if debug:
            dbg = {"crop": crop, "comp": np.clip(comp, 0, 255).astype(np.uint8), "alpha": alpha, "eu": eu, "el": el, "score": score, "delta": self.delta.copy()}
        return out, dbg
