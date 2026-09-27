import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "teeth_model.npz")
SEG_PATH = os.path.join(HERE, "teeth_seg.onnx")
STATE_PATH = os.path.join(HERE, "teeth_memory.npz")
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
    if v.size < k:
        return np.asarray(v, np.float32).copy()
    return cv2.medianBlur(np.ascontiguousarray(v, np.float32).reshape(1, -1), k).ravel()


class AnchorFill:
    def __init__(self, model_path=MODEL_PATH, seg_path=SEG_PATH, offset=None, axis_shift=0.0):
        z = np.load(model_path)
        self.df = DentalFrame(z["ref_shape"].astype(np.float64), z["rigid_idx"])
        self.x_mid = float(z["x_mid"]) + axis_shift
        self.bottom = z["bottom"].astype(np.float32)
        self.slot = z["slot"].astype(np.float32) * np.clip((GX - (float(z["x_mid"]) - 66.0)) / 10.0, 0, 1)
        self.donor_mask = z["donor_mask"] > 0.5
        xs = np.arange(CW)
        mir = np.interp(2 * self.x_mid - xs, xs, self.bottom, left=0, right=0)
        full = np.where(xs < self.x_mid, mir, self.bottom)
        good = full > 0
        self.bottom_full = smooth_cols(np.interp(xs, xs[good], full[good]), 5).astype(np.float32)
        self.body_top = np.argmax(self.slot > 0.5, 0)
        ys, xs = np.nonzero(self.slot > 0.05)
        self.roi = (max(int(ys.min()) - 2, 0), min(int(ys.max()) + 4, CH), max(int(xs.min()) - 2, 0), min(int(np.ceil(self.x_mid)) + 7, CW))
        band = (np.abs(GX - self.x_mid) <= 5) & (GY >= self.roi[0]) & (GY < self.roi[1])
        self.slot_lim = (self.slot > 0.05) | band
        self.xd_map = np.ascontiguousarray(np.broadcast_to((2 * self.x_mid - GX[0])[None, :], (CH, CW))).astype(np.float32)
        self.seg = Segmenter(seg_path)
        self.edge_x0 = float(z["x_mid"]) + float(z["edge_offset"])
        self.upper = True
        self.lower = True
        self.extend_px = 6.0
        self.bottom_margin = 1.0
        self.smooth = "crop"
        self.k_new = 0.5
        self.k_reg = 0.35
        self.use_reg = False
        self.robust = True
        self.robust_n = 12
        self.med_pull = 0.15
        self.max_step = 1.5
        self.y_clip = 3.0
        self.y_gain = 0.4
        self.live_ext = 6.0
        self.relock_conf = 0.12
        self.l_need = 1
        self.lower_axis = True
        self.l_bridge = 15
        self.occl = True
        self.l_rows = 22
        self.lax_n = 0
        self.l_inset = 0.0
        self.l_src_fix = True
        self.l_edge_fill = 14
        self.l_hold = 6
        self.l_reach_left = 34
        self.l_min_rows = 10
        self.l_need_gap = 0.15
        self.fall_l_none = 0.34
        self.l_near = 12
        self.l_span = 76
        self.l_fade = 0.0
        self.l_lip_shade = 0.25
        self.l_feather = 6.0
        self.l_light = 0.0
        self.l_slit = 0.3
        self.light_cols = (68, 90)
        self.lax_step = 0.5
        self.rise_l = 1.0
        self.stable = True
        self.st_rate = 0.12
        self.shade = False
        self.gain_lo, self.gain_hi, self.gain_k = 0.4, 1.8, 0.7
        self.rise_u = 1.0
        self.seam_band = True
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
        self.mem = None
        self.mem_w = None
        self.mem_bot = None
        self.au_prev = None
        self.al_prev = None
        self.lmem = None
        self.l_hits = 0
        self.prev_hp = None
        self.prev_mouth = None
        self.reg_ok = False
        self.recent = []
        self.lax = None
        self.lax_hist = []
        self.light_rl = 1.0
        self.lcm_t = None
        self.lcm_b = None
        self.lcm_age = None
        self.relock_t = 0
        self.st_tex = None
        self.st_seen = None
        self.st_sil = None
        self.st_gain = None
        self.st_shade = None
        self.st_gap = 99

    def save_state(self, path=STATE_PATH):
        if self.st_tex is None or (self.st_seen > 0.3).sum() < 200:
            return False
        np.savez_compressed(path, st_tex=self.st_tex.astype(np.float16), st_seen=self.st_seen.astype(np.float16),
                            st_sil=self.st_sil.astype(np.float16), roi=np.array(self.roi))
        return True

    def load_state(self, path=STATE_PATH):
        if not os.path.exists(path):
            return False
        z = np.load(path)
        if tuple(int(v) for v in z["roi"]) != tuple(self.roi):
            return False
        self.st_tex = z["st_tex"].astype(np.float32)
        self.st_seen = np.minimum(z["st_seen"].astype(np.float32), 0.6)
        self.st_sil = z["st_sil"].astype(np.float32)
        self.st_gain = np.ones(3, np.float32)
        return True

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

    def register(self, hp, mouth0):
        if self.prev_hp is None or self.delta is None:
            return None
        loc = np.array([self.et.x0, self.et.b0]) + self.delta
        x0 = int(round(loc[0])) - 50
        x1 = int(round(loc[0])) + 50
        y1 = int(round(loc[1])) + 2
        y0 = y1 - 30
        p = 4
        if x0 - p < 0 or x1 + p > CW or y0 - p < 0 or y1 + p > CH:
            return None
        m = (self.prev_mouth[y0:y1, x0:x1] > 0.7) & (mouth0[y0:y1, x0:x1] > 0.7)
        pl = self.prev_L[y0:y1, x0:x1]
        if m.sum() > 60:
            m &= pl > np.percentile(pl[m], 40)
        m = m.astype(np.float32)
        if m.sum() < 120:
            return None
        tpl = self.prev_hp[y0:y1, x0:x1]
        if tpl[m > 0].std() < 3.0:
            return None
        win = hp[y0 - p:y1 + p, x0 - p:x1 + p]
        r = cv2.matchTemplate(win, tpl, cv2.TM_CCORR_NORMED, mask=m)
        r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
        _, mv, _, ml = cv2.minMaxLoc(r)
        if mv < 0.6:
            return None
        px, py = ml
        if px in (0, r.shape[1] - 1) or py in (0, r.shape[0] - 1):
            return None
        return np.array([px + parabolic(r[py], px) - p, py + parabolic(r[:, px], py) - p])

    def anchor(self, a0, L0, mouth0):
        hp = L0 - cv2.GaussianBlur(L0, (0, 0), 3.0)
        shift = self.register(hp, mouth0) if self.use_reg else None
        self.prev_hp, self.prev_mouth, self.prev_L = hp, mouth0, L0
        meas = self.et.measure(L0, mouth0)
        base = np.array([self.et.x0, self.et.b0])
        pred = None
        if self.delta is not None:
            pred = self.delta + shift if shift is not None else self.delta.copy()
        ref = pred if (pred is not None and (self.miss < 30 or shift is not None)) else self.et.off
        med = float(np.median(self.recent)) if (self.robust and len(self.recent) >= 5) else None
        good = False
        if meas is not None and meas["conf"] >= 0.06:
            yo = self.et.off[1]
            dy = float(np.clip((meas["y"] - base[1]) - yo, -self.y_clip, self.y_clip)) if meas["y"] is not None else 0.0
            d_m = np.array([meas["x"] - base[0], yo + self.y_gain * dy])
            if self.robust and meas["conf"] >= 0.12:
                self.recent = (self.recent + [d_m[0]])[-self.robust_n:]
            err = abs(d_m[0] - ref[0])
            if med is not None:
                err = min(err, abs(d_m[0] - med) + 1.0)
            if self.et.n < 15:
                good = meas["conf"] >= 0.12
            elif meas["conf"] >= 0.12 and err <= min(4.0 + 0.25 * self.miss, 12.0):
                good = True
            elif err <= 2.5:
                good = True
            elif meas["conf"] >= self.relock_conf:
                self.outliers.append(d_m[0])
                good = len(self.outliers) >= 3 and np.ptp(self.outliers[-3:]) < 4
                if good and self.robust:
                    self.recent = list(self.outliers[-3:])
                    self.relock_t = 6
            if good:
                self.outliers = []
        self.reg_ok = shift is not None
        if good:
            self.et.update(meas)
            self.miss = 0
            self.locks += 1
            if pred is None:
                self.delta = d_m
            else:
                k = self.k_reg if shift is not None else self.k_new
                self.delta = k * d_m + (1 - k) * pred
        else:
            self.miss += 1
            if pred is None:
                self.delta = self.et.off.copy()
            else:
                self.delta = pred
                if med is not None:
                    self.delta = self.delta.copy()
                    self.delta[0] += self.med_pull * (med - self.delta[0])
                elif shift is None and self.miss > 8:
                    self.delta = self.delta + 0.05 * (self.et.off - self.delta)
        if pred is not None and self.max_step > 0:
            prev = pred - (shift if shift is not None else 0)
            step = self.max_step * (2.0 if self.relock_t > 0 else 1.0)
            self.delta[0] = prev[0] + float(np.clip(self.delta[0] - prev[0], -step, step))
        self.relock_t = max(self.relock_t - 1, 0)

    def hold(self, raw, prev, limit, rise=0.6, fall=0.34):
        if prev is None:
            a = raw * rise
        else:
            a = prev + np.clip(raw - prev, -fall, rise)
        a = np.minimum(a, limit).astype(np.float32)
        return a, a

    def upper_fill(self, crop, L, mouth):
        r0, r1, c0, c1 = self.roi
        R = (slice(r0, r1), slice(c0, c1))
        h = r1 - r0
        body = self.slot[R]
        inb = body > 0.5
        gy = GY[R]
        rows = np.arange(r0, r1, dtype=np.float32)[:, None]
        lc = np.arange(c1 - c0)
        xm = self.xd_map[R]
        mouth_d = cv2.remap(mouth, xm, gy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        L_d = cv2.remap(L, xm, gy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        live = cv2.remap(crop, xm, gy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
        vis_d = (mouth_d > 0.65) & inb
        has = vis_d.any(0)
        top_d = np.maximum(np.where(has, r0 + np.argmax(vis_d, 0), CH), self.body_top[c0:c1]).astype(np.float32)
        lt = float(np.percentile(L_d[vis_d], 75)) if vis_d.sum() > 20 else 0.0
        bright = vis_d & (L_d > 0.62 * lt)
        bf = self.bottom_full[c0:c1]
        cap = bf + self.bottom_margin
        start = np.maximum(top_d + 2, bf - 10)
        at_start = bright[np.clip(start - r0, 0, h - 1).astype(np.int32), lc]
        below = (rows >= start[None, :]) & ~bright
        first_dark = np.where(below.any(0), r0 + np.argmax(below, 0), CH).astype(np.float32)
        above = bright & (rows <= start[None, :]) & (rows >= top_d[None, :])
        last_up = np.where(above.any(0), r0 + h - np.argmax(above[::-1], 0), top_d).astype(np.float32)
        bot = np.minimum(np.where(at_start, first_dark, last_up), cap)
        sel = np.flatnonzero(has)
        if sel.size >= 5:
            b = cv2.dilate(bot[sel].astype(np.float32).reshape(1, -1), np.ones((1, 9), np.uint8)).ravel()
            bot[sel] = np.minimum(smooth_cols(b, 5), cap[sel])
        live_ok = has & (bot > top_d + 2)
        raw_bot = np.where(at_start, first_dark, last_up).astype(np.float32)
        if self.stable:
            vis_s = (mouth_d > 0.75) & inb
            self.stable_update(live, vis_s, vis_s & (L_d > 0.62 * lt) if lt > 0 else np.zeros_like(vis_s))
            st = self.stable_render(rows, gy, body, cap)
            if st is not None:
                tex, cover, has_st = st
                if self.live_ext > 0:
                    tex, cover = self.extend_live(tex, cover, live, raw_bot, cap + self.live_ext, has & has_st, rows, gy)
                mt = self.extend_dark(mouth[R], L[R], lt, inb)
                self.mouth_t = mouth.copy()
                self.mouth_t[R] = mt
                alpha = body * np.clip((mt - 0.3) / 0.4, 0, 1) * cover
                tex, alpha = self.heal_seam(crop[R], tex, alpha, lt, self.x_mid - c0)
                tex = np.where((alpha > 0)[..., None], tex, crop[R].astype(np.float32))
                tex_full = crop.astype(np.float32)
                tex_full[R] = tex
                alpha_full = np.zeros((CH, CW), np.float32)
                alpha_full[R] = cv2.GaussianBlur(alpha.astype(np.float32), (0, 0), 0.7)
                return tex_full, alpha_full
        mem_ok = self.update_memory(live, mouth_d, vis_d, top_d, bot, live_ok, rows)
        top_b = np.clip(np.where(bright.any(0), r0 + np.argmax(bright, 0), top_d), top_d, top_d + 6).astype(np.float32)
        live_src = (rows >= top_b[None, :]) & live_ok[None, :]
        use_mem = ~live_src & mem_ok
        src_ok = live_src | use_mem
        tex = np.where(use_mem[..., None], self.mem, live)
        has_src = src_ok.any(0)
        t_src = np.where(has_src, r0 + np.argmax(src_ok, 0), CH).astype(np.int32)
        samp = np.stack([tex[np.clip(t_src + k - r0, 0, h - 1), lc] for k in (3, 4, 5, 6)])
        top_px = np.median(samp, 0).astype(np.float32)
        top_px = cv2.GaussianBlur(top_px.reshape(1, -1, 3), (0, 0), sigmaX=2.0, sigmaY=0.1).reshape(-1, 3)
        dist = np.clip((t_src[None, :] - gy) / 14.0, 0, 1)
        ext = top_px[None, :, :] * (1 - 0.22 * dist)[..., None]
        need_ext = (rows < t_src[None, :]) & has_src[None, :]
        tex = np.where(need_ext[..., None], ext, tex).astype(np.float32)
        bot_eff = np.where(live_ok, bot, np.where(np.isnan(self.mem_bot), 0, self.mem_bot)).astype(np.float32)
        mt = self.extend_dark(mouth[R], L[R], lt, inb)
        self.mouth_t = mouth.copy()
        self.mouth_t[R] = mt
        alpha = body * np.clip((mt - 0.3) / 0.4, 0, 1) * np.clip(bot_eff[None, :] + 0.5 - gy, 0, 1)
        alpha *= (has_src & (bot_eff > t_src + 2))[None, :]
        tex, alpha = self.heal_seam(crop[R], tex, alpha, lt, self.x_mid - c0)
        tex_full = crop.astype(np.float32)
        tex_full[R] = tex
        alpha_full = np.zeros((CH, CW), np.float32)
        alpha_full[R] = cv2.GaussianBlur(alpha.astype(np.float32), (0, 0), 0.7)
        return tex_full, alpha_full

    def stable_update(self, live, vis, tooth_now):
        if self.st_tex is None:
            self.st_tex = live.copy()
            self.st_seen = np.zeros(live.shape[:2], np.float32)
            self.st_sil = np.zeros(live.shape[:2], np.float32)
            self.st_gain = np.ones(3, np.float32)
        both = vis & tooth_now & (self.st_seen > 0.5)
        if both.sum() > 40:
            g = np.array([np.median(live[..., c][both] / np.maximum(self.st_tex[..., c][both], 1.0)) for c in range(3)], np.float32)
            g = np.clip(g, self.gain_lo, self.gain_hi)
            k = 1.0 if self.st_gap > 2 else self.gain_k
            self.st_gain = ((1 - k) * self.st_gain + k * g).astype(np.float32)
            self.st_gap = 0
        else:
            self.st_gap += 1
        if self.shade:
            self.shade_update(live, vis & (self.st_seen > 0.5) & (self.st_sil > 0.5))
        if vis.sum() > 20:
            norm = live / self.st_gain[None, None, :]
            a = np.where(self.st_seen <= 0.01, 1.0, self.st_rate)[..., None]
            upd = tooth_now | (vis & (self.st_seen <= 0.01))
            self.st_tex = np.where(upd[..., None], (1 - a) * self.st_tex + a * norm, self.st_tex).astype(np.float32)
            self.st_seen = np.where(vis, self.st_seen + 0.25 * (1 - self.st_seen), self.st_seen).astype(np.float32)
            self.st_sil = np.where(vis, self.st_sil + 0.2 * (tooth_now.astype(np.float32) - self.st_sil), self.st_sil).astype(np.float32)

    def shade_update(self, live, both):
        if self.st_shade is None:
            self.st_shade = np.ones(live.shape[:2], np.float32)
        if both.sum() <= 40:
            self.st_shade = (0.8 * self.st_shade + 0.2).astype(np.float32)
            return
        w = both.astype(np.float32)
        ratio = live.mean(2) / np.maximum((self.st_tex * self.st_gain[None, None, :]).mean(2), 1.0)
        num = cv2.GaussianBlur(np.clip(ratio, 0.25, 1.5) * w, (0, 0), 6.0)
        den = cv2.GaussianBlur(w, (0, 0), 6.0)
        conf = np.clip(den / 0.2, 0, 1)
        new = 1 + (num / np.maximum(den, 1e-6) - 1) * conf
        self.st_shade = (0.5 * self.st_shade + 0.5 * np.clip(new, 0.35, 1.3)).astype(np.float32)

    def extend_live(self, tex, cover, live, raw_bot, cap2, ok, rows, gy):
        sb = np.where((cover > 0.5).any(0), rows[-1, 0] - np.argmax((cover > 0.5)[::-1], 0), 0).astype(np.float32)
        lb = np.minimum(raw_bot, cap2)
        sel = np.flatnonzero(ok)
        if sel.size >= 5:
            lb[sel] = smooth_cols(lb[sel], 5)
        grow = ok & (lb > sb + 1)
        if not grow.any():
            return tex, cover
        band = grow[None, :] & (rows > sb[None, :]) & (rows <= lb[None, :])
        tex = np.where(band[..., None], live, tex).astype(np.float32)
        cover = np.maximum(cover, band * np.clip(lb[None, :] + 0.5 - gy, 0, 1)).astype(np.float32)
        return tex, cover

    def stable_render(self, rows, gy, body, cap):
        h = rows.shape[0]
        r0 = int(rows[0, 0])
        known = self.st_seen > 0.3
        tooth = known & (self.st_sil > 0.5)
        tooth = cv2.morphologyEx(tooth.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 3), np.uint8)) > 0
        has = tooth.any(0)
        if has.sum() < 15:
            return None
        top = np.where(has, r0 + np.argmax(tooth, 0), CH).astype(np.float32)
        bot = np.where(has, r0 + h - np.argmax(tooth[::-1], 0), 0).astype(np.float32)
        sel = np.flatnonzero(has)
        b = cv2.dilate(bot[sel].reshape(1, -1), np.ones((1, 9), np.uint8)).ravel()
        bot[sel] = np.minimum(smooth_cols(b, 5), cap[sel])
        t = -cv2.dilate(-top[sel].reshape(1, -1), np.ones((1, 9), np.uint8)).ravel()
        top[sel] = smooth_cols(t, 5)
        tex = self.st_tex * self.st_gain[None, None, :]
        if self.shade and self.st_shade is not None:
            tex = tex * self.st_shade[..., None]
        lc = np.arange(tex.shape[1])
        ti = np.clip(top.astype(np.int32) - r0, 0, h - 1)
        samp = np.stack([tex[np.clip(ti + k, 0, h - 1), lc] for k in (2, 3, 4, 5)])
        top_px = cv2.GaussianBlur(np.median(samp, 0).astype(np.float32).reshape(1, -1, 3), (0, 0), sigmaX=2.0, sigmaY=0.1).reshape(-1, 3)
        dist = np.clip((top[None, :] - gy) / 14.0, 0, 1)
        need = (rows < top[None, :] + 2) & has[None, :]
        ext = top_px[None, :, :] * (1 - 0.22 * dist)[..., None]
        blend = np.clip((top[None, :] + 2 - rows) / 3.0, 0, 1)[..., None] * need[..., None]
        tex = (tex * (1 - blend) + ext * blend).astype(np.float32)
        cover = has[None, :] & (rows <= bot[None, :])
        return tex, cover.astype(np.float32) * np.clip(bot[None, :] + 0.5 - gy, 0, 1), has

    def extend_dark(self, mouth, L, lt, inb, reach=12):
        if lt <= 0:
            return mouth.copy()
        hh, ww = mouth.shape
        conf = (mouth > 0.5) & inb
        has = conf.sum(0) >= 4
        top = np.where(has, np.argmax(conf, 0), hh)
        dark = L < 0.55 * lt
        out = mouth.copy()
        alive = has.copy()
        cols = np.arange(ww)
        n_ext = np.zeros(ww, np.int32)
        for k in range(1, reach + 1):
            r = top - k
            ok = alive & (r >= 0)
            rr = np.clip(r, 0, hh - 1)
            ok &= dark[rr, cols] & inb[rr, cols]
            n_ext[ok] = k
            alive = ok
        ext_cols = n_ext > 0
        if ext_cols.any():
            rows = np.arange(hh)[:, None]
            fill = ext_cols[None, :] & (rows >= (top - n_ext)[None, :]) & (rows <= (top + 6)[None, :])
            out = np.where(fill, np.maximum(out, 0.9), out).astype(np.float32)
        return out

    def update_memory(self, live, mouth_d, vis_d, top_d, bot, live_ok, rows):
        crown = vis_d & (mouth_d > 0.8) & (rows >= top_d[None, :]) & (rows < bot[None, :]) & live_ok[None, :]
        if self.mem is None:
            self.mem = live.copy()
            self.mem_w = np.zeros(live.shape[:2], np.float32)
            self.mem_bot = np.full(live.shape[1], np.nan, np.float32)
        both = crown & (self.mem_w > 0.5)
        if both.sum() > 40:
            g = np.array([np.median(live[..., c][both] / np.maximum(self.mem[..., c][both], 1.0)) for c in range(3)], np.float32)
            self.mem *= np.clip(g, 0.85, 1.18)[None, None, :]
        if crown.sum() > 30:
            fresh = crown & (self.mem_w <= 0.05)
            self.mem = np.where(fresh[..., None], live, np.where(crown[..., None], 0.5 * self.mem + 0.5 * live, self.mem)).astype(np.float32)
            self.mem_w = np.where(crown, 1.0, self.mem_w * 0.997).astype(np.float32)
            cols = live_ok & (bot > top_d + 4)
            prev = self.mem_bot
            self.mem_bot = np.where(cols, np.where(np.isnan(prev), bot, 0.7 * prev + 0.3 * bot), prev).astype(np.float32)
        else:
            self.mem_w *= 0.997
        return (self.mem_w > 0.35) & (rows < np.nan_to_num(self.mem_bot, nan=0)[None, :])

    def heal_seam(self, crop, tex, alpha, lt, axis):
        a = int(round(axis))
        x0, x1 = a - 4, a + 5
        if x0 < 0 or x1 >= crop.shape[1] or lt <= 0:
            return tex, alpha
        comp = crop[:, x0:x1].astype(np.float32) * (1 - alpha[:, x0:x1, None]) + tex[:, x0:x1] * alpha[:, x0:x1, None]
        left, right = comp[:, 0], comp[:, -1]
        lum = comp.mean(2)
        ok = (alpha[:, x0 + 1] > 0.5) & (left.mean(1) > 0.55 * lt) & (right.mean(1) > 0.55 * lt)
        if not ok.any():
            return tex, alpha
        xs = np.arange(x0, x1, dtype=np.float32)
        w = (xs - x0) / (x1 - 1 - x0)
        lin = left[:, None, :] * (1 - w)[None, :, None] + right[:, None, :] * w[None, :, None]
        target = lin * (1 - 0.2 * np.exp(-((xs - axis) / 0.9) ** 2))[None, :, None]
        k = np.clip((target.mean(2) - lum) / 15.0, 0, 1)[..., None]
        healed = comp + (target - comp) * k
        tex = tex.copy()
        alpha = alpha.copy()
        tex[ok, x0:x1] = healed[ok]
        alpha[ok, x0:x1] = np.maximum(alpha[ok, x0:x1], alpha[ok, x0 + 1:x0 + 2])
        return tex, alpha

    def find_lower_axis(self, L, inm, thr, reach=20):
        c0, c1 = int(self.x_mid) - self.l_reach_left - 8, int(self.x_mid) + reach + 6
        cols = np.arange(c0, c1)
        rows = np.arange(CH)[:, None]
        has = inm[:, cols].any(0)
        lb = np.where(has, CH - 1 - np.argmax(inm[::-1, cols], 0), -1)
        ub = self.bottom_full[cols] + 1.0
        band = inm[:, cols] & (rows > ub[None, :]) & (rows <= lb[None, :]) & (rows >= lb[None, :] - self.l_rows)
        n = band.sum(0)
        if band.sum() >= 40:
            vals = L[:, cols][band]
            hi, lo = float(np.percentile(vals, 90)), float(np.percentile(vals, 15))
            if hi - lo > 25:
                thr = 0.5 * (hi + lo)
        p = np.where(n >= 4, ((L[:, cols] >= thr) & band).sum(0) / np.maximum(n, 1), 0.0)
        cs = np.concatenate([[0.0], np.cumsum(p)])
        j = np.arange(8, len(cols) - 6)
        left = (cs[j + 1] - cs[j - 7]) / 8
        right = (cs[j + 6] - cs[j + 1]) / 5
        q = (left > 0.4) & (right < 0.3) & (left - right > 0.25)
        found = cols[j[np.argmax(q)]] + 0.5 - self.l_inset if q.any() and np.median(n[has]) >= self.l_min_rows else None
        self.l_thr = thr
        if self.lax_n > 0:
            if found is not None:
                self.lax_hist = (self.lax_hist + [float(found)])[-self.lax_n:]
            if self.lax_hist:
                target = float(np.median(self.lax_hist))
                if self.lax is None:
                    self.lax = target
                else:
                    self.lax += float(np.clip(target - self.lax, -self.lax_step, self.lax_step))
        elif found is not None:
            if self.lax is None:
                self.lax = float(found)
            else:
                self.lax += float(np.clip(0.6 * (found - self.lax), -3.0, 3.0))
        return self.lax if self.lax is not None else self.x_mid

    def lower_col_memory(self, t_s, b_s, valid, xs, xm, span):
        if self.lcm_t is None:
            self.lcm_t = np.full(CW, np.nan, np.float32)
            self.lcm_b = np.full(CW, np.nan, np.float32)
            self.lcm_age = np.full(CW, 99, np.int32)
        both = valid & ~np.isnan(self.lcm_t)
        dy = float(np.median(t_s[both] - self.lcm_t[both])) if both.sum() >= 5 else 0.0
        keep = ~valid & (self.lcm_age < self.l_hold) & ~np.isnan(self.lcm_t) & (xs > xm) & (xs < xm + span)
        t_s = np.where(keep, self.lcm_t + dy, t_s)
        b_s = np.where(keep, self.lcm_b + dy, b_s)
        self.lcm_t = np.where(valid, t_s, self.lcm_t + dy).astype(np.float32)
        self.lcm_b = np.where(valid, b_s, self.lcm_b + dy).astype(np.float32)
        self.lcm_age = np.where(valid, 0, self.lcm_age + 1).astype(np.int32)
        return t_s, b_s, valid | keep

    def update_light(self, L, mouth):
        xm = int(self.x_mid)
        inm = mouth > 0.5
        ry = np.arange(CH)[:, None]
        ub = np.where(inm.any(0), np.argmax(inm, 0), CH)
        up = inm & (ry < self.bottom_full[None, :]) & (ry > ub[None, :])
        a0, a1 = self.light_cols
        lm, rm = up[:, xm - a1:xm - a0], up[:, xm + a0:xm + a1]
        if lm.sum() > 30 and rm.sum() > 30:
            r = float(np.percentile(L[:, xm + a0:xm + a1][rm], 85) / max(np.percentile(L[:, xm - a1:xm - a0][lm], 85), 1.0))
            self.light_rl += 0.1 * (float(np.clip(r, 0.5, 1.2)) - self.light_rl)

    def lower_fill(self, crop, L, mouth, span=64):
        xm = self.x_mid
        inm = mouth > 0.5
        cols = np.arange(int(xm - span), int(xm))
        vis_ref = self.donor_mask & (mouth > 0.7)
        low_reg = inm[:, cols] & (np.arange(CH)[:, None] > self.bottom_full[cols][None, :] + 1)
        if vis_ref.sum() >= 20:
            lt = float(np.percentile(L[vis_ref], 70))
            thr = 0.72 * lt
        elif low_reg.sum() >= 60:
            vals = L[:, cols][low_reg]
            thr = 0.5 * (float(np.percentile(vals, 92)) + float(np.percentile(vals, 30)))
            lt = float(np.percentile(vals, 92))
        else:
            return None
        if self.lower_axis:
            xm = self.find_lower_axis(L, inm, thr)
            thr = self.l_thr
            cols = np.arange(int(xm - span), int(xm))
        if self.l_need_gap > 0:
            tc = np.arange(int(xm) + 2, min(int(xm) + 30, CW))
            tin = inm[:, tc]
            tlb = np.where(tin.any(0), CH - 1 - np.argmax(tin[::-1], 0), -1)
            ry = np.arange(CH)[:, None]
            tband = tin & (ry > self.bottom_full[tc][None, :] + 1) & (ry <= tlb[None, :])
            if tband.sum() >= 30 and inm[:, cols].any():
                hi = float(np.percentile(L[:, cols][inm[:, cols]], 90))
                if (L[:, tc][tband] < 0.7 * hi).mean() < self.l_need_gap:
                    return None
        has = inm[:, cols].any(0)
        lb = np.where(has, CH - 1 - np.argmax(inm[::-1, cols], 0), -1)
        ub = self.bottom_full[cols] + 1.0
        rows = np.arange(CH)[:, None]
        cand = (L[:, cols] >= thr) & inm[:, cols] & (rows <= lb[None, :]) & (rows > ub[None, :])
        near = rows >= lb[None, :] - self.l_near
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
        t = -cv2.dilate(-t.reshape(1, -1), np.ones((1, self.l_bridge), np.uint8)).ravel()
        okb = t < 1e4
        t[okb] = smooth_cols(t[okb], 5)
        b = np.where(ok, bot, 0).astype(np.float32)
        b = cv2.dilate(b.reshape(1, -1), np.ones((1, self.l_bridge), np.uint8)).ravel()
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
        if self.l_edge_fill > 0:
            tgt = np.flatnonzero(valid)
            x0i = int(np.floor(xm)) + 1
            if tgt.size and x0i < tgt[0] <= x0i + self.l_edge_fill:
                fv = tgt[0]
                t_s[x0i:fv] = t_s[fv]
                b_s[x0i:fv] = b_s[fv]
                valid[x0i:fv] = True
        if self.l_hold > 0:
            t_s, b_s, valid = self.lower_col_memory(t_s, b_s, valid, xs, xm, span)
        low = np.maximum(np.where(valid, t_s, CH) - 0.5, self.bottom_full + 1.0)
        feather = np.ones(CW, np.float32)
        if self.l_feather > 0 and valid.any():
            v8 = np.pad(valid.astype(np.uint8), 1).reshape(1, -1)
            dist = cv2.distanceTransform(np.ascontiguousarray(v8), cv2.DIST_L1, 3).ravel()[1:-1]
            right = xs > xm + self.l_edge_fill + 2
            feather = np.where(right, np.clip(dist / self.l_feather, 0, 1), 1.0).astype(np.float32)
        taper = np.clip((xm + span - xs) / 10.0, 0, 1) * np.clip((xs - xm) / 1.5, 0, 1)
        xa, xb = max(int(xm) - 10, 0), min(int(np.ceil(xm + span)) + 4, CW)
        ya = max(int(np.floor(low[xa:xb].min())) - 4, 0)
        if ya >= CH - 1:
            return None
        B = (slice(ya, CH), slice(xa, xb))
        alpha_b = np.clip((GY[B] - low[None, xa:xb]) / 1.5, 0, 1) * np.clip((mouth[B] - 0.15) / 0.3, 0, 1) * (valid * taper * feather)[None, xa:xb]
        mapy = np.minimum(GY[B], np.nan_to_num(b_s[xa:xb], nan=CH)[None, :]).astype(np.float32)
        xd_row = 2 * xm - xs[xa:xb].astype(np.float32)
        if self.l_src_fix and (~ok).any():
            okidx = np.flatnonzero(ok)
            k = np.arange(len(cols))
            pos = np.clip(np.searchsorted(okidx, k), 1, len(okidx) - 1) if len(okidx) > 1 else np.zeros(len(k), int)
            lo_i, hi_i = okidx[np.maximum(pos - 1, 0)], okidx[np.minimum(pos, len(okidx) - 1)]
            near = np.where(np.abs(k - lo_i) <= np.abs(hi_i - k), lo_i, hi_i)
            shift = np.zeros(CW, np.float32)
            shift[cols] = (cols[near] - cols).astype(np.float32)
            di = np.clip(np.round(xd_row).astype(int), 0, CW - 1)
            xd_row = xd_row + shift[di]
        xd = np.ascontiguousarray(np.broadcast_to(xd_row[None, :], (CH - ya, xb - xa))).astype(np.float32)
        tex_b = cv2.remap(crop, xd, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
        if self.l_fade > 0 or self.l_lip_shade > 0 or self.l_light > 0 or self.l_slit > 0:
            shade = 1 - self.l_fade * np.clip((xs[xa:xb] - xm) / span, 0, 1)[None, :]
            if self.l_light > 0:
                shade = shade * min(1.0, max(self.light_rl, 0.6)) ** self.l_light
            if self.l_slit > 0:
                ob = mouth[B] > 0.5
                h_t = ob.sum(0).astype(np.float32)
                h_d = np.nan_to_num(b_s[xa:xb] - t_s[xa:xb], nan=0.0).astype(np.float32)
                slit = np.where(h_d > 3, np.clip(h_t / np.maximum(h_d, 1) , 0, 1), 1.0)
                shade = shade * (1 - self.l_slit * (1 - slit))[None, :]
            on = alpha_b > 0.5
            bot_r = np.where(on.any(0), on.shape[0] - 1 - np.argmax(on[::-1], 0), -1)
            ry = np.arange(on.shape[0])[:, None]
            near_lip = np.clip(1 - (bot_r[None, :] - ry) / 4.0, 0, 1) * (bot_r >= 0)[None, :]
            shade = shade * (1 - self.l_lip_shade * near_lip)
            tex_b = tex_b * shade[..., None]
        crop_b = crop[B].astype(np.float32)
        if self.seam_band:
            t2, a2 = self.heal_seam(crop_b[:, ::-1], tex_b[:, ::-1], alpha_b[:, ::-1].astype(np.float32), lt, (xb - xa - 1) - (xm - xa))
            tex_b, alpha_b = t2[:, ::-1], a2[:, ::-1]
            tex_b = np.where((alpha_b > 0)[..., None], tex_b, crop_b)
        tex = crop.astype(np.float32)
        tex[B] = tex_b
        alpha = np.zeros((CH, CW), np.float32)
        alpha[B] = cv2.GaussianBlur(alpha_b.astype(np.float32), (0, 0), 0.8, borderType=cv2.BORDER_CONSTANT)
        return tex, alpha

    def occlusion_guard(self, tex, L, alpha):
        ys, xs = np.nonzero(alpha > 0.01)
        y0, y1, x0, x1 = max(ys.min() - 3, 0), min(ys.max() + 4, CH), max(xs.min() - 3, 0), min(xs.max() + 4, CW)
        lt = cv2.cvtColor(np.clip(tex[y0:y1, x0:x1], 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
        g = 1 - np.clip((L[y0:y1, x0:x1] - lt - 4) / 12.0, 0, 1)
        alpha = alpha.copy()
        alpha[y0:y1, x0:x1] *= cv2.GaussianBlur(g, (0, 0), 1.2)
        return alpha

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
        gate = float(np.clip(self.locks / 3.0, 0, 1))
        mclip = np.clip((mouth - 0.3) / 0.4, 0, 1)
        L = cv2.GaussianBlur(cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32), (0, 0), 0.7)
        tex = crop.astype(np.float32)
        alpha = np.zeros((CH, CW), np.float32)
        alpha_u = alpha
        alpha_l = alpha
        if self.upper and gate > 0:
            tex_u, raw_u = self.upper_fill(crop, L, mouth)
            lim_slot = self.slot_lim if self.seam_band else (self.slot > 0.05)
            alpha_u, self.au_prev = self.hold(raw_u * gate, self.au_prev, np.clip((self.mouth_t - 0.3) / 0.4, 0, 1) * lim_slot, rise=self.rise_u)
            if self.mem is not None and not self.stable:
                r0, r1, c0, c1 = self.roi
                held = (alpha_u[r0:r1, c0:c1] > raw_u[r0:r1, c0:c1] + 0.05) & (self.mem_w > 0.35)
                tex_u[r0:r1, c0:c1] = np.where(held[..., None], self.mem, tex_u[r0:r1, c0:c1])
            tex, alpha = tex_u, alpha_u
        else:
            self.au_prev = None
        if self.lower and gate > 0:
            if self.l_light > 0:
                self.update_light(L, mouth)
            lf = self.lower_fill(crop, L, mouth, span=self.l_span)
            raw_l = np.zeros((CH, CW), np.float32)
            if lf is not None:
                tex_l, raw_l = lf
                self.l_hits += 1
                if self.lmem is None:
                    self.lmem = tex_l.copy()
                upd = raw_l > 0.5
                self.lmem[upd] = tex_l[upd]
            else:
                self.l_hits = 0
                tex_l = self.lmem if self.lmem is not None else tex
            if self.l_hits < self.l_need:
                raw_l = raw_l * 0.0
            lim_l = mclip
            if self.lower_axis and self.lax is not None:
                lim_l = mclip * (GX >= self.lax - 4)
            alpha_l, self.al_prev = self.hold(raw_l * gate, self.al_prev, lim_l, rise=self.rise_l, fall=0.34 if lf is not None else self.fall_l_none)
            if self.lmem is not None:
                tex_l = np.where((alpha_l > raw_l + 0.05)[..., None], self.lmem, tex_l)
            use_l = alpha_l > alpha
            tex = np.where(use_l[..., None], tex_l, tex)
            alpha = np.maximum(alpha, alpha_l)
        else:
            self.al_prev = None
        if self.occl and alpha.max() > 0.01:
            alpha = self.occlusion_guard(tex, L, alpha)
        self.last = (a, alpha)
        if self.debug:
            self.dbg = {"crop": crop, "mouth": mouth, "alpha": alpha, "tex": tex, "delta": self.delta.copy(), "miss": self.miss, "locks": self.locks, "gate": gate, "alpha_u": alpha_u, "alpha_l": alpha_l}
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
