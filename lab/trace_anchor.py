import sys, cv2, numpy as np
sys.path.insert(0, "..")
import dental as D
key = sys.argv[1]; n_max = int(sys.argv[2]) if len(sys.argv) > 2 else 10 ** 9
d = dict(np.load(f"cache/{key}.npz", allow_pickle=True))
cap = cv2.VideoCapture(str(d["path"]))
af = D.AnchorFill(offset=[-0.14, -13.71])
for k, v in (a.split("=") for a in sys.argv[3:]):
    setattr(af, k, eval(v))
orig_measure = af.et.measure
last = {}
def spy(L, mouth):
    m = orig_measure(L, mouth)
    last["m"] = m
    return m
af.et.measure = spy
orig_reg = af.register
def spy_reg(hp, mouth0):
    s = orig_reg(hp, mouth0)
    last["s"] = s
    return s
af.register = spy_reg
n = 0
while n < n_max:
    ok, f = cap.read()
    if not ok: break
    last.clear()
    if d["ok"][n]:
        af.process(f, d["pts"][n].astype(np.float64))
        m = last.get("m"); s = last.get("s")
        mx = f"{m['x'] - af.et.x0:6.1f} c{m['conf']:.2f} y{(m['y'] - af.et.b0 - af.et.off[1]) if m['y'] is not None else float('nan'):6.1f}" if m else "   -        "
        sx = f"{s[0]:5.1f}" if s is not None else "  -  "
        print(f"{n:4d} delta {af.delta[0]:6.1f} {af.delta[1]:6.1f} meas {mx} reg {sx} off {af.et.off[0]:5.1f} miss {af.miss}")
    n += 1
