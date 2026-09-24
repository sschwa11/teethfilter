import sys, cv2, numpy as np
sys.path.insert(0, "..")
import dental as D


def hp(g):
    g = g.astype(np.float32)
    return g - cv2.GaussianBlur(g, (0, 0), 3)


def evaluate(keys, setup, hide_period=40, hide_len=10):
    jit, herr, meas = [], [], 0
    for key in keys:
        d = dict(np.load(f"cache/{key}.npz", allow_pickle=True))
        cap = cv2.VideoCapture(str(d["path"]))
        af = D.AnchorFill(); af.debug = True; setup(af)
        orig = af.et.measure
        state = {"hide": False, "hidden": None}
        def spy(L, mouth):
            m = orig(L, mouth)
            if state["hide"] and m is not None and m["conf"] >= 0.12:
                state["hidden"] = m["x"] - af.et.x0
                return None
            state["hidden"] = None
            return m
        af.et.measure = spy
        prev = None
        n = 0
        while True:
            ok, f = cap.read()
            if not ok: break
            if d["ok"][n]:
                state["hide"] = (n % hide_period) >= hide_period - hide_len
                af.dbg = None
                af.process(f, d["pts"][n].astype(np.float64))
                if af.miss == 0: meas += 1
                if state["hidden"] is not None and af.delta is not None:
                    herr.append(abs(af.delta[0] - state["hidden"]))
                g = af.dbg
                if g is not None:
                    reg = (af.donor_mask & (g["mouth"] > 0.8))
                    gray = hp(cv2.cvtColor(g["crop"], cv2.COLOR_BGR2GRAY))
                    if prev is not None and reg.sum() > 150 and prev[1].sum() > 150 and not state["hide"]:
                        m = (reg & prev[1]).astype(np.float32)
                        if m.sum() < 100:
                            prev = (gray, reg); n += 1; continue
                        ys, xs = np.nonzero(m)
                        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
                        if y1 - y0 > 6 and x1 - x0 > 16:
                            a = prev[0][y0:y1, x0:x1]
                            b = gray[max(y0 - 4, 0):y1 + 4, max(x0 - 4, 0):x1 + 4]
                            r = cv2.matchTemplate(b, a, cv2.TM_CCOEFF_NORMED)
                            _, mv, _, ml = cv2.minMaxLoc(r)
                            if mv > 0.6:
                                jit.append(abs(ml[0] - min(4, x0)))
                    prev = (gray, reg)
                else:
                    prev = None
            else:
                prev = None
            n += 1
    jit, herr = np.array(jit), np.array(herr)
    return {"frame_jitter_x": round(float(jit.mean()), 3), "jit>1": round(float((jit >= 1).mean()), 3), "n_jit": len(jit), "hidden_mae": round(float(herr.mean()), 2), "hidden_p90": round(float(np.percentile(herr, 90)), 2), "measured": meas}


if __name__ == "__main__":
    keys = ["2026-09-23_05-47-17_raw", "2026-09-23_05-46-00_raw", "2026-09-22_23-36-51_raw"]
    configs = {
        "image_oneeuro": lambda af: setattr(af, "smooth", "image"),
        "crop_k0.7": lambda af: (setattr(af, "smooth", "crop"), setattr(af, "k_new", 0.7)),
        "crop_k1.0": lambda af: (setattr(af, "smooth", "crop"), setattr(af, "k_new", 1.0)),
        "crop_k0.4": lambda af: (setattr(af, "smooth", "crop"), setattr(af, "k_new", 0.4)),
    }
    for name, fn in configs.items():
        print(name, evaluate(keys, fn), flush=True)
