import sys, cv2, numpy as np, common as C, render_v10 as V
key = sys.argv[1]; frames = [int(x) for x in sys.argv[2].split(",")]
out_name = sys.argv[3] if len(sys.argv) > 3 else "dbg.jpg"
d = np.load(f"cache/{key}.npz", allow_pickle=True)
cap = cv2.VideoCapture(str(d["path"]))
r = V.Renderer(V.Model())
rows = []
n = 0
last = max(frames)
while n <= last:
    ok, f = cap.read()
    if not ok: break
    if d["ok"][n]:
        out, dbg = r.process(f, d["pts"][n].astype(np.float64), debug=True)
        if n in frames and dbg is not None:
            crop = dbg["crop"].copy(); comp = dbg["comp"]
            for x in range(60, 260):
                for key_, col in (("eu", (0, 255, 0)), ("el", (255, 0, 0)), ("top_s", (255, 0, 255)), ("bot_s", (255, 255, 0))):
                    v = dbg.get(key_)
                    if v is not None and 0 < v[x] < 259:
                        cv2.circle(crop, (x, int(v[x])), 0, col, -1)
                b = r.m.bottom[x]
                if b > 0: cv2.circle(crop, (x, int(b)), 0, (0, 0, 255), -1)
            cv2.line(crop, (int(r.m.x_mid), 90), (int(r.m.x_mid), 170), (0, 255, 255), 1)
            al = cv2.cvtColor((dbg["alpha"] * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            t = np.hstack([crop[85:185, 40:280], comp[85:185, 40:280], al[85:185, 40:280]])
            t = cv2.resize(t, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
            cv2.putText(t, f"{n} score {dbg['score']:.2f} d {dbg['delta'].round(1)}", (4, 16), 0, 0.5, (0, 255, 0), 1)
            rows.append(t)
    n += 1
cv2.imwrite("out/" + out_name, np.vstack(rows))
