import json
import sys

import cv2
import numpy as np

sys.path.insert(0, "..")
import dental as D

CLIPS = ["2026-09-25_08-03-41_raw", "2026-09-25_02-36-08_raw", "2026-09-25_01-18-25_raw", "2026-09-25_01-10-45_raw"]
IDX = [14, 17, 18, 200, 199, 175, 152, 87, 317, 86, 316, 84, 314, 13, 0]


def collect(key):
    d = dict(np.load(f"cache/{key}.npz", allow_pickle=True))
    cap = cv2.VideoCapture(str(d["path"]))
    af = D.AnchorFill()
    af.reset(json.load(open("../settings.json"))["anchor_offset"])
    af.upper = False
    orig = af.find_lower_axis
    st = {}

    def spy(L, inm, thr, reach=20):
        before = af.lax
        af.lax = None
        r = orig(L, inm, thr, reach)
        st["found"] = af.lax
        af.lax = before if before is not None else r
        return af.lax

    af.find_lower_axis = spy
    rows = []
    n = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        st.clear()
        if d["ok"][n]:
            pts = d["pts"][n].astype(np.float64)
            af.process(f, pts)
            if st.get("found") is not None and af.last is not None:
                cp = D.to_crop(af.last[0], pts)
                rows.append([n, st["found"]] + [cp[i, 0] for i in IDX] + [cp[i, 1] for i in (14, 152, 17)])
        n += 1
    return np.array(rows)


if __name__ == "__main__":
    allr = {}
    for k in CLIPS:
        allr[k] = collect(k)
        print(k, len(allr[k]), flush=True)
    np.savez("out/jaw_fit.npz", **{k: v for k, v in allr.items()})
