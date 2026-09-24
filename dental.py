import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "teeth_model.npz")
SEG_PATH = os.path.join(HERE, "teeth_seg.onnx")
CW, CH = 320, 260
SEG_ROWS = (2, 258)
GY, GX = np.mgrid[0:CH, 0:CW].astype(np.float32)
UPPER_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
LOWER_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
INNER_POLY = UPPER_INNER + LOWER_INNER[::-1][1:-1]


def umeyama(src, dst):
    ms, md = src.mean(0), dst.mean(0)
    a, b = src - ms, dst - md
    u, d, vt = np.linalg.svd(b.T @ a / len(src))
    s_ = np.ones(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s_[-1] = -1
    r = u @ np.diag(s_) @ vt
    s = (d * s_).sum() / ((a ** 2).sum() / len(src))
    return s, r, md - s * r @ ms


class DentalFrame:
    def __init__(self, ref, idx):
        ex = ref[263] - ref[33]
        ex /= np.linalg.norm(ex)
        ey = ref[152] - ref[10]
        ey -= ex * np.dot(ey, ex)
        ey /= np.linalg.norm(ey)
        self.ex, self.ey = ex, ey
        self.ref = ref
        self.idx = np.asarray(idx)
        ez = np.cross(ex, ey)
        self.center = (ref[13] + ref[14]) / 2 + 0.08 * np.linalg.norm(ref[263] - ref[33]) * ez
        self.k = np.linalg.norm(ref[308] - ref[78]) / 200.0

    def pose(self, pts):
        return umeyama(self.ref[self.idx], pts[self.idx].astype(np.float64))

    def affine(self, pose):
        s, r, t = pose
        cols = np.stack([self.ex * self.k, self.ey * self.k], 1)
        o = self.center - (CW / 2) * self.k * self.ex - (CH / 2) * self.k * self.ey
        a = np.zeros((2, 3))
        a[:, :2] = (s * r @ cols)[:2]
        a[:, 2] = (s * r @ o + t)[:2]
        return a


def to_crop(a, pts):
    inv = np.linalg.inv(np.vstack([a, [0, 0, 1]]))
    return pts[:, :2] @ inv[:2, :2].T + inv[:2, 2]


class OneEuro:
    def __init__(self, min_cutoff=1.0, beta=0.02, dcutoff=1.0, fps=30.0):
        self.mc, self.beta, self.dc, self.fps = min_cutoff, beta, dcutoff, fps
        self.x = None
        self.dx = None

    def alpha(self, cutoff):
        return 1.0 / (1.0 + self.fps / (2 * np.pi * cutoff))

    def __call__(self, x):
        if self.x is None:
            self.x, self.dx = x.copy(), np.zeros_like(x)
            return self.x
        ad = self.alpha(self.dc)
        self.dx = ad * (x - self.x) * self.fps + (1 - ad) * self.dx
        a = self.alpha(self.mc + self.beta * np.abs(self.dx))
        self.x = a * x + (1 - a) * self.x
        return self.x

    def reset(self, x):
        if self.x is not None:
            self.x = x.copy()
            self.dx = np.zeros_like(x)


def parabolic(v, i):
    if 0 < i < len(v) - 1:
        den = v[i - 1] - 2 * v[i] + v[i + 1]
        if abs(den) > 1e-6:
            return 0.5 * (v[i - 1] - v[i + 1]) / den
    return 0.0


class EdgeTracker:
    def __init__(self, x0, b0, crown=24, search=12, width=30):
        self.x0, self.b0 = x0, b0
        self.crown, self.search, self.width = crown, search, width
        self.off = np.zeros(2)
        self.n = 0

    def measure(self, L, mouth):
        xp = self.x0 + self.off[0]
        bp = self.b0 + self.off[1]
        r0, r1 = int(max(bp - self.crown, 0)), int(min(bp - 2, CH))
        c0, c1 = int(max(xp - self.search - 4, 0)), int(min(xp + self.search + self.width, CW))
        if r1 - r0 < 3 or c1 - c0 < 20:
            return None
        vis = mouth[r0:r1, c0:c1] > 0.6
        j0 = int(xp) - c0
        rows = vis[:, j0:j0 + 16].mean(1) > 0.7
        if rows.sum() < 3:
            return None
        prof = L[r0:r1, c0:c1][rows].mean(0)
        lo_t, hi_t = np.percentile(prof, 10), np.percentile(prof, 90)
        contrast = hi_t - lo_t
        if contrast < 15:
            return None
        k = 2
        e = np.zeros_like(prof)
        e[k:-k] = prof[2 * k:] - prof[:-2 * k]
        lo, hi = max(int(xp - self.search) - c0, k), min(int(xp + self.search) - c0, len(e) - k)
        if hi <= lo:
            return None
        seg = e[lo:hi]
        cs = np.concatenate([[0.0], np.cumsum(prof)])
        idx = lo + np.arange(len(seg))
        tw = 26
        r_end = np.minimum(idx + 2 + tw, len(prof))
        l_beg = np.maximum(idx - 9, 0)
        plateau = (cs[r_end] - cs[np.minimum(idx + 2, len(prof))]) / np.maximum(r_end - idx - 2, 1)
        gapm = (cs[np.maximum(idx - 2, 1)] - cs[l_beg]) / np.maximum(idx - 2 - l_beg, 1)
        box = np.clip(plateau - gapm, 0, None)
        score = np.clip(seg, 0, None) / max(seg.max(), 1e-6) + box / max(box.max(), 1e-6)
        sigma = 5.0 if self.n >= 15 else 20.0
        xs = c0 + idx
        j = int(np.argmax(score * np.exp(-0.5 * ((xs - xp) / sigma) ** 2)))
        xe = c0 + lo + j + parabolic(seg, j)
        right = prof[min(lo + j + 4, len(prof) - 1):min(lo + j + 20, len(prof))]
        left = prof[max(lo + j - 8, 0):max(lo + j - 3, 1)]
        if right.size < 4 or left.size < 2 or right.mean() - left.mean() < 0.35 * contrast:
            return None
        conf = float(seg[j] / (2 * k) / max(contrast, 1))
        cols = slice(int(xe + 4), int(xe + 4 + min(24, self.width - 6)))
        vr0, vr1 = int(max(bp - (24 if self.n < 40 else 12), 0)), int(min(bp + 10, CH - 1))
        ye = None
        q = L[vr0:vr1, cols].mean(1)
        if q.size > 6:
            dq = np.zeros_like(q)
            dq[1:-1] = q[:-2] - q[2:]
            dq[mouth[vr0:vr1, cols].mean(1) < 0.6] = 0
            i = int(np.argmax(dq))
            if dq[i] > 0.25 * contrast:
                ye = vr0 + i + parabolic(dq, i) + 0.5
        return {"x": float(xe), "y": ye, "conf": conf}

    def update(self, meas, rate=0.1):
        self.n += 1
        r = max(rate, 1.0 / self.n)
        self.off[0] += r * ((meas["x"] - self.x0) - self.off[0])
        if meas["y"] is not None:
            dy = (meas["y"] - self.b0) - self.off[1]
            if self.n < 40 or abs(dy) < 6:
                self.off[1] += (r if self.n < 40 else 0.03) * dy


class Segmenter:
    def __init__(self, path=SEG_PATH):
        import onnxruntime as ort
        avail = ort.get_available_providers()
        providers = [p for p in ("DmlExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider") if p in avail]
        self.sess = ort.InferenceSession(path, providers=providers)
        self.provider = self.sess.get_providers()[0]
        self.name = self.sess.get_inputs()[0].name

    def __call__(self, crop):
        y0, y1 = SEG_ROWS
        x = crop[y0:y1][None].astype(np.float32)
        out = np.zeros((CH, CW), np.float32)
        out[y0:y1] = self.sess.run(None, {self.name: x})[0][0]
        return out


def shift_img(img, d, border=cv2.BORDER_REPLICATE):
    m = np.float32([[1, 0, d[0]], [0, 1, d[1]]])
    return cv2.warpAffine(img, m, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=border, borderValue=0)


def smooth_cols(v, k=5):
    pad = np.pad(v, k // 2, mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(pad, k), 1)


class AnchorFill:
    def __init__(self, model_path=MODEL_PATH, seg_path=SEG_PATH, offset=None):
        z = np.load(model_path)
        self.df = DentalFrame(z["ref_shape"].astype(np.float64), z["rigid_idx"])
        self.x_mid = float(z["x_mid"])
        self.bottom = z["bottom"].astype(np.float32)
        self.slot = z["slot"].astype(np.float32) * np.clip((GX - (float(z["x_mid"]) - 66.0)) / 10.0, 0, 1)
        self.donor_mask = z["donor_mask"] > 0.5
        xs = np.arange(CW)
        mir = np.interp(2 * self.x_mid - xs, xs, self.bottom, left=0, right=0)
        full = np.where(xs < self.x_mid, mir, self.bottom)
        good = full > 0
        self.bottom_full = np.interp(xs, xs[good], full[good]).astype(np.float32)
        self.body_top = np.argmax(self.slot > 0.5, 0)
        self.xd_map = np.ascontiguousarray(np.broadcast_to((2 * self.x_mid - GX[0])[None, :], (CH, CW))).astype(np.float32)
        self.seg = Segmenter(seg_path)
        self.edge_x0 = self.x_mid + float(z["edge_offset"])
        self.upper = True
        self.lower = True
        self.extend_px = 6.0
        self.bottom_margin = 1.0
        self.smooth = "crop"
        self.k_new = 0.7
        self.debug = False
        self.dbg = None
        self.reset(offset)

    def reset(self, offset=None):
        self.et = EdgeTracker(self.edge_x0, float(self.bottom[int(self.x_mid) + 18]))
        if offset is not None:
            self.et.off = np.asarray(offset, np.float64).copy()
            self.et.n = 15
        self.filt = OneEuro(1.0, 0.02)
        self.delta = None
        self.miss = 99
        self.locks = 3 if offset is not None else 0
        self.outliers = []
        self.alpha_prev = None
        self.alpha_l_prev = None
        self.last = None

    @property
    def offset(self):
        return self.et.off.copy()

    def clean_mouth(self, mouth, cp):
        band = np.zeros((CH, CW), np.uint8)
        cv2.fillPoly(band, [np.round(cp[INNER_POLY]).astype(np.int32)], 1)
        band = cv2.dilate(band, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
        m = mouth * cv2.GaussianBlur(band.astype(np.float32), (0, 0), 2.0)
        n, labels, stats, _ = cv2.connectedComponentsWithStats((m > 0.5).astype(np.uint8), connectivity=8)
        if n > 2:
            keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            m = m * cv2.dilate((labels == keep).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(np.float32)
        return m

    def anchor(self, a0, L0, mouth0):
        meas = self.et.measure(L0, mouth0)
        base = np.array([self.et.x0, self.et.b0])
        ref = self.delta if (self.delta is not None and self.miss < 30) else self.et.off
        good = False
        if meas is not None and meas["conf"] >= 0.06:
            yo = self.et.off[1]
            dy = float(np.clip((meas["y"] - base[1]) - yo, -3, 3)) if meas["y"] is not None else 0.0
            d_m = np.array([meas["x"] - base[0], yo + 0.4 * dy])
            err = abs(d_m[0] - ref[0])
            if self.et.n < 15:
                good = meas["conf"] >= 0.12
            elif meas["conf"] >= 0.12 and err <= min(4.0 + 0.25 * self.miss, 12.0):
                good = True
            elif err <= 2.5:
                good = True
            elif meas["conf"] >= 0.12:
                self.outliers.append(d_m[0])
                good = len(self.outliers) >= 3 and np.ptp(self.outliers[-3:]) < 4
            if good:
                self.outliers = []
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
        if self.smooth == "image":
            p = a0[:, :2] @ loc + a0[:, 2]
            if not good:
                self.filt.reset(p)
            pf = self.filt(p)
            inv = np.linalg.inv(np.vstack([a0, [0, 0, 1]]))
            self.delta = inv[:2, :2] @ pf + inv[:2, 2] - base
        else:
            new = loc - base
            if good and self.delta is not None and self.miss == 0:
                new = self.k_new * new + (1 - self.k_new) * self.delta
            self.delta = new

    def upper_fill(self, crop, L, mouth):
        body = self.slot
        mouth_d = cv2.remap(mouth, self.xd_map, GY, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        L_d = cv2.remap(L, self.xd_map, GY, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        vis_d = (mouth_d > 0.65) & (body > 0.5)
        has = vis_d.any(0)
        top_d = np.maximum(np.where(has, np.argmax(vis_d, 0), CH), self.body_top).astype(np.float32)
        tooth = vis_d & (body > 0.5)
        lt = float(np.percentile(L_d[tooth], 75)) if tooth.sum() > 20 else 0.0
        bright = tooth & (L_d > 0.62 * lt)
        rows = np.arange(CH)[:, None]
        cap = self.bottom_full + self.bottom_margin
        lowb = bright & (rows >= top_d[None, :]) & (rows <= cap[None, :])
        last_b = np.where(lowb.any(0), CH - 1 - np.argmax(lowb[::-1], 0) + 1.0, top_d).astype(np.float32)
        bot = np.minimum(last_b, cap)
        sel = np.flatnonzero(has)
        if sel.size >= 5:
            b = cv2.dilate(bot[sel].astype(np.float32).reshape(1, -1), np.ones((1, 9), np.uint8)).ravel()
            bot[sel] = np.minimum(smooth_cols(b, 5), self.bottom_full[sel] + self.bottom_margin)
        top_b = np.where(bright.any(0), np.argmax(bright, 0), top_d).astype(np.float32)
        top_b = np.clip(top_b, top_d, top_d + 6)
        mapy = np.maximum(GY, (top_b + 1.0)[None, :]).astype(np.float32)
        tex = cv2.remap(crop, self.xd_map, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
        alpha = body * np.clip((mouth - 0.3) / 0.4, 0, 1)
        alpha *= np.clip(bot[None, :] + 0.5 - GY, 0, 1)
        alpha *= np.clip((GY - (top_d[None, :] - self.extend_px)) / 3.0, 0, 1)
        alpha *= (has & (bot > top_d + 2))[None, :]
        return tex, cv2.GaussianBlur(alpha.astype(np.float32), (0, 0), 0.7)

    def lower_fill(self, crop, L, mouth, span=64):
        xm = self.x_mid
        inm = mouth > 0.5
        cols = np.arange(int(xm - span), int(xm))
        vis_ref = self.donor_mask & (mouth > 0.7)
        low_reg = inm[:, cols] & (np.arange(CH)[:, None] > self.bottom_full[cols][None, :] + 1)
        if vis_ref.sum() >= 20:
            thr = 0.72 * float(np.percentile(L[vis_ref], 70))
        elif low_reg.sum() >= 60:
            vals = L[:, cols][low_reg]
            thr = 0.5 * (float(np.percentile(vals, 92)) + float(np.percentile(vals, 30)))
        else:
            return None
        has = inm[:, cols].any(0)
        lb = np.where(has, CH - 1 - np.argmax(inm[::-1, cols], 0), -1)
        ub = self.bottom_full[cols] + 1.0
        rows = np.arange(CH)[:, None]
        cand = (L[:, cols] >= thr) & inm[:, cols] & (rows <= lb[None, :]) & (rows > ub[None, :])
        near = rows >= lb[None, :] - 6
        start_ok = (cand & near).any(0)
        bot = np.where(start_ok, CH - 1 - np.argmax((cand & near)[::-1], 0), -1)
        gap = ~cand & (rows <= bot[None, :]) & (rows > ub[None, :])
        last_gap = np.where(gap.any(0), CH - 1 - np.argmax(gap[::-1], 0), ub.astype(int))
        top = last_gap + 1
        ok = start_ok & has & (bot - top >= 4)
        if ok.sum() < 12:
            return None
        t = np.where(ok, top, 0).astype(np.float32)
        t = np.where(t > 0, t, 1e4).astype(np.float32)
        t = -cv2.dilate(-t.reshape(1, -1), np.ones((1, 7), np.uint8)).ravel()
        okb = t < 1e4
        t[okb] = smooth_cols(t[okb], 5)
        b = np.where(ok, bot, 0).astype(np.float32)
        b = cv2.dilate(b.reshape(1, -1), np.ones((1, 7), np.uint8)).ravel()
        top_l = np.full(CW, np.nan, np.float32)
        bot_l = np.full(CW, np.nan, np.float32)
        top_l[cols[okb]] = t[okb]
        bot_l[cols[okb]] = b[okb]
        xs = np.arange(CW)
        xdi = np.clip(np.round(2 * xm - xs).astype(int), 0, CW - 1)
        in_span = (xs > xm) & (xs < xm + span)
        t_s = np.where(in_span, top_l[xdi], np.nan)
        b_s = np.where(in_span, bot_l[xdi], np.nan)
        valid = ~np.isnan(t_s)
        low = np.maximum(np.where(valid, t_s, CH) - 0.5, self.bottom_full + 1.0)
        taper = np.clip((xm + span - xs) / 10.0, 0, 1) * np.clip((xs - xm) / 1.5, 0, 1)
        alpha = np.clip((GY - low[None, :]) / 1.5, 0, 1) * np.clip((mouth - 0.15) / 0.3, 0, 1) * (valid * taper)[None, :]
        mapy = np.minimum(GY, np.nan_to_num(b_s, nan=CH)[None, :]).astype(np.float32)
        tex = cv2.remap(crop, self.xd_map, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
        return tex, cv2.GaussianBlur(alpha.astype(np.float32), (0, 0), 0.8)

    def process(self, frame, pts):
        pose = self.df.pose(pts)
        a0 = self.df.affine(pose)
        flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
        crop0 = cv2.warpAffine(frame, a0.astype(np.float32), (CW, CH), flags=flags, borderMode=cv2.BORDER_REPLICATE)
        cp0 = to_crop(a0, pts)
        mouth0 = self.clean_mouth(self.seg(crop0), cp0)
        L0 = cv2.GaussianBlur(cv2.cvtColor(crop0, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32), (0, 0), 0.8)
        self.anchor(a0, L0, mouth0)
        a = a0.copy()
        a[:, 2] += a0[:, :2] @ self.delta
        crop = cv2.warpAffine(frame, a.astype(np.float32), (CW, CH), flags=flags, borderMode=cv2.BORDER_REPLICATE)
        mouth = shift_img(mouth0, self.delta, cv2.BORDER_CONSTANT)
        cp = cp0 - self.delta
        yu = np.interp(np.arange(CW), *zip(*sorted(cp[UPPER_INNER].tolist())))
        yl = np.interp(np.arange(CW), *zip(*sorted(cp[LOWER_INNER].tolist())))
        xm = int(self.x_mid)
        opening = float(np.median((yl - yu)[xm - 40:xm + 40]))
        gate = float(np.clip((opening - 2) / 5, 0, 1)) * float(np.clip(self.locks / 3.0, 0, 1))
        tex = crop.astype(np.float32)
        alpha = np.zeros((CH, CW), np.float32)
        L = cv2.GaussianBlur(cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32), (0, 0), 0.7)
        if self.upper and gate > 0:
            tex, alpha = self.upper_fill(crop, L, mouth)
            alpha *= gate
            if self.alpha_prev is not None:
                alpha = np.maximum(alpha, 0.35 * self.alpha_prev * (mouth > 0.5))
        self.alpha_prev = alpha if self.upper else None
        if self.lower and gate > 0:
            lf = self.lower_fill(crop, L, mouth)
            if lf is not None:
                tex_l, alpha_l = lf
                alpha_l *= gate
                if self.alpha_l_prev is not None:
                    alpha_l = 0.6 * alpha_l + 0.4 * self.alpha_l_prev * (mouth > 0.5)
                self.alpha_l_prev = alpha_l
                use_l = alpha_l > alpha
                tex = np.where(use_l[..., None], tex_l, tex)
                alpha = np.maximum(alpha, alpha_l)
            else:
                self.alpha_l_prev = None
        self.last = (a, alpha)
        if self.debug:
            self.dbg = {"crop": crop, "mouth": mouth, "alpha": alpha, "tex": tex, "delta": self.delta.copy(), "miss": self.miss, "locks": self.locks, "gate": gate}
        if alpha.max() < 0.01:
            return frame, None
        return self.paste(frame, pts, a, tex, alpha)

    def paste(self, frame, pts, a, tex, alpha):
        ys, xs = np.nonzero(alpha > 0.004)
        if ys.size == 0:
            return frame, None
        cy0, cy1 = max(int(ys.min()) - 2, 0), min(int(ys.max()) + 3, CH)
        cx0, cx1 = max(int(xs.min()) - 2, 0), min(int(xs.max()) + 3, CW)
        corners = np.array([[cx0, cy0], [cx1, cy0], [cx0, cy1], [cx1, cy1]], np.float64)
        img = corners @ a[:, :2].T + a[:, 2]
        bx0, by0 = int(max(np.floor(img[:, 0].min()) - 1, 0)), int(max(np.floor(img[:, 1].min()) - 1, 0))
        bx1, by1 = int(min(np.ceil(img[:, 0].max()) + 2, frame.shape[1])), int(min(np.ceil(img[:, 1].max()) + 2, frame.shape[0]))
        if bx1 - bx0 < 2 or by1 - by0 < 2:
            return frame, None
        sh = np.array([[1, 0, bx0], [0, 1, by0], [0, 0, 1]], np.float64)
        fwd = (np.linalg.inv(np.vstack([a, [0, 0, 1]])) @ sh)[:2].astype(np.float32)
        size = (bx1 - bx0, by1 - by0)
        flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
        rep = cv2.warpAffine(tex, fwd, size, flags=flags, borderMode=cv2.BORDER_REPLICATE)
        ab = cv2.warpAffine(alpha, fwd, size, flags=flags, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        out = frame.copy()
        roi = out[by0:by1, bx0:bx1].astype(np.float32)
        out[by0:by1, bx0:bx1] = np.clip(roi + (rep - roi) * ab[..., None], 0, 255).astype(np.uint8)
        return out, (bx0, by0, ab)


def available():
    return os.path.exists(MODEL_PATH) and os.path.exists(SEG_PATH)
