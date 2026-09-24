import collections
import json
import sys

res_file, tag = sys.argv[1], sys.argv[2]
t = open(res_file, encoding="utf-8").read()
res = json.loads(t[t.find("["):]) if t.lstrip().startswith("[") else json.loads(t[t.find("{"):])["result"]
key = {k["file"]: k for k in json.load(open(f"out/judge_{tag}/key.json"))}
new = tag.split("_")[-1]
g = collections.defaultdict(list)
per = collections.defaultdict(list)
notes = []
for judge in res:
    for r in judge["ratings"]:
        k = key[r["file"]]
        sn = "A" if k["A"] == new else "B"
        so = "B" if sn == "A" else "A"
        g["flick"].append(r["less_flicker"])
        g["nat"].append(r["more_natural"])
        g["f_new"].append(r["flicker_" + sn])
        g["f_old"].append(r["flicker_" + so])
        per[r["file"]].append(r["flicker_" + so] - r["flicker_" + sn])
        notes.append((r["file"], sn, r["notes"]))
n = len(g["flick"])
lf = collections.Counter("new" if v != "tie" and v == ("A" if key[f]["A"] == new else "B") else ("tie" if v == "tie" else "live") for v, f in zip(g["flick"], [x[0] for x in notes]))
mn = collections.Counter("new" if v != "tie" and v == ("A" if key[f]["A"] == new else "B") else ("tie" if v == "tie" else "live") for v, f in zip(g["nat"], [x[0] for x in notes]))
print(f"{tag}: n={n} less_flicker new/live/tie {lf['new']}/{lf['live']}/{lf['tie']}  more_natural {mn['new']}/{mn['live']}/{mn['tie']}  flicker new {sum(g['f_new'])/n:.2f} vs live {sum(g['f_old'])/n:.2f}")
for f in sorted(per):
    print(f, key[f]["clip"][11:19], key[f]["start"], "live-new", per[f])
if len(sys.argv) > 3:
    for f, sn, nt in notes:
        print(f, "new=" + sn, "|", nt[:300])
