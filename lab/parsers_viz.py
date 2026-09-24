import cv2, numpy as np, common as C, parsers as P
df = C.dental()
ps = {k: P.Parser(k) for k in ("segformer", "bisenet")}
tiles = []
sel = (("2026-09-23_05-46-00_raw", [37, 74, 111, 150, 185, 230, 400]), ("2026-09-22_23-36-22_raw", [30, 200, 300]), ("2026-09-22_23-02-18", [148, 600, 820, 900]))
for key, frames in sel:
    d = np.load(f"cache/{key}.npz", allow_pickle=True)
    cap = cv2.VideoCapture(str(d["path"]))
    n = 0
    while n <= max(frames):
        ok, f = cap.read()
        if n in frames and d["ok"][n]:
            pts = d["pts"][n].astype(np.float64)
            a = df.affine(df.pose(pts))
            crop = cv2.warpAffine(f, a.astype(np.float32), (320, 260), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
            row = []
            for name, col in (("segformer", (0, 255, 0)), ("bisenet", (255, 0, 255))):
                p, m, fc = ps[name].probs(f, pts)
                dd = ps[name].to_dental(p, m, a)
                big = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                mm = cv2.resize((dd["mouth"] > 0.5).astype(np.uint8), None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
                cs, _ = cv2.findContours(mm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(big, cs, -1, col, 1)
                lu = cv2.resize((dd["u_lip"] > 0.5).astype(np.uint8), None, fx=2, fy=2)
                cs, _ = cv2.findContours(lu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(big, cs, -1, (0, 128, 255), 1)
                ll = cv2.resize((dd["l_lip"] > 0.5).astype(np.uint8), None, fx=2, fy=2)
                cs, _ = cv2.findContours(ll, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(big, cs, -1, (255, 128, 0), 1)
                cv2.putText(big, f"{name} {key[11:19]} {n}", (4, 14), 0, 0.45, col, 1)
                row.append(big[160:400, 80:560])
            tiles.append(np.hstack(row))
        n += 1
cv2.imwrite("out/parsers_a.jpg", np.vstack(tiles[:7]))
cv2.imwrite("out/parsers_b.jpg", np.vstack(tiles[7:]))
