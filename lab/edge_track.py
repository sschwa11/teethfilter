import cv2
import numpy as np


def parabolic(v, i):
    if 0 < i < len(v) - 1:
        den = v[i - 1] - 2 * v[i] + v[i + 1]
        if abs(den) > 1e-6:
            return 0.5 * (v[i - 1] - v[i + 1]) / den
    return 0.0


class EdgeTracker:
    def __init__(self, x_mid=157.5, bottom=137.0, crown=24, search=12, width=30):
        self.x0 = x_mid
        self.b0 = bottom
        self.crown = crown
        self.search = search
        self.width = width
        self.off = np.zeros(2)
        self.n = 0

    def measure(self, crop, mouth):
        L = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
        L = cv2.GaussianBlur(L, (0, 0), 0.8)
        xp = self.x0 + self.off[0]
        bp = self.b0 + self.off[1]
        r0 = int(max(bp - self.crown, 0))
        r1 = int(min(bp - 2, L.shape[0]))
        c0 = int(max(xp - self.search - 4, 0))
        c1 = int(min(xp + self.search + self.width, L.shape[1]))
        vis = mouth[r0:r1, c0:c1] > 0.6
        rows = vis[:, int(xp) - c0:int(xp) - c0 + 16].mean(1) > 0.7 if int(xp) - c0 + 16 <= vis.shape[1] else np.zeros(r1 - r0, bool)
        if rows.sum() < 3:
            return None
        band = L[r0:r1, c0:c1][rows]
        prof = band.mean(0)
        lo_t = np.percentile(prof, 10)
        hi_t = np.percentile(prof, 90)
        contrast = hi_t - lo_t
        if contrast < 15:
            return None
        k = 2
        e = np.zeros_like(prof)
        e[k:-k] = prof[2 * k:] - prof[:-2 * k]
        lo = int(xp - self.search) - c0
        hi = int(xp + self.search) - c0
        lo, hi = max(lo, k), min(hi, len(e) - k)
        if hi <= lo:
            return None
        seg = e[lo:hi]
        j = int(np.argmax(seg))
        emax = seg[j]
        xe = c0 + lo + j + parabolic(seg, j)
        right = prof[min(lo + j + 4, len(prof) - 1):min(lo + j + 20, len(prof))]
        left = prof[max(lo + j - 8, 0):max(lo + j - 3, 1)]
        if right.size < 4 or left.size < 2:
            return None
        conf = float(emax / (2 * k) / max(contrast, 1))
        tooth_bright = right.mean() - left.mean()
        if tooth_bright < 0.35 * contrast:
            return None
        cols = slice(int(xe + 4), int(xe + 4 + min(24, self.width - 6)))
        vr0 = int(max(bp - (24 if self.n < 40 else 12), 0))
        vr1 = int(min(bp + 10, L.shape[0] - 1))
        q = L[vr0:vr1, cols].mean(1)
        mq = mouth[vr0:vr1, cols].mean(1)
        ye = None
        if q.size > 6:
            dq = np.zeros_like(q)
            dq[1:-1] = q[:-2] - q[2:]
            dq[mq < 0.6] = 0
            i = int(np.argmax(dq))
            if dq[i] > 0.25 * contrast:
                ye = vr0 + i + parabolic(dq, i) + 0.5
        return {"x": float(xe), "y": ye, "conf": conf, "contrast": float(contrast)}

    def update(self, meas, rate=0.1):
        if meas is None or meas["conf"] < 0.12:
            return
        self.n += 1
        r = max(rate, 1.0 / self.n)
        self.off[0] += r * ((meas["x"] - self.x0) - self.off[0])
        if meas["y"] is not None:
            self.off[1] += r * 0.5 * ((meas["y"] - self.b0) - self.off[1])
