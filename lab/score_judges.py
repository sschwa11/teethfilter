import json, sys, collections
out_file, tag = sys.argv[1], sys.argv[2]
t = open(out_file, encoding="utf-8").read()
res = json.loads(t[t.find("{"):])["result"]
key = {k["file"]: k for k in json.load(open(f"out/judge_{tag}/key.json"))}
agg = collections.defaultdict(lambda: collections.defaultdict(list))
arts = []
for judge in res:
    for r in judge["ratings"]:
        k = key[r["file"]]
        sn = "A" if k["A"] == tag else "B"; so = "B" if sn == "A" else "A"
        other = k[so]
        g = agg[other]
        g["win"].append(1 if r["more_natural"] == sn else (0 if r["more_natural"] == so else 0.5))
        g["nat_new"].append(r["natural_" + sn]); g["nat_old"].append(r["natural_" + so])
        g["gap_new"].append(r["gap_visible_" + sn]); g["gap_old"].append(r["gap_visible_" + so])
        arts.append((r["file"], k["clip"][11:19], k["frame"], other, r["artifacts_" + sn], r["natural_" + sn]))
for other, g in agg.items():
    n = len(g["win"])
    print(f"{tag} vs {other}: n={n} win={sum(g['win'])/n:.2f} natural {sum(g['nat_new'])/n:.2f} vs {sum(g['nat_old'])/n:.2f}  gap visible {sum(g['gap_new'])/n:.2f} vs {sum(g['gap_old'])/n:.2f}")
if len(sys.argv) > 3:
    for a in sorted(arts, key=lambda a: a[5])[:14]:
        print(a[5], a[0], a[1], a[2], "|", a[4][:200])
