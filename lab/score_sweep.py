import collections
import json
import sys

res_file, name = sys.argv[1], sys.argv[2]
t = open(res_file, encoding="utf-8").read()
res = json.loads(t[t.find("{"):])["result"] if not t.lstrip().startswith("[") else json.loads(t)
index = {e["file"]: e["start"] for e in json.load(open(f"out/sweep_{name}/index.json"))}
by_strip = collections.defaultdict(list)
for r in res:
    for s in r["strips"]:
        by_strip[s["file"]].append(s)
cats = collections.defaultdict(list)
clean_both = 0
reviewed = 0
for f, reps in sorted(by_strip.items()):
    if len(reps) < 2:
        continue
    reviewed += 1
    if all(x["clean"] for x in reps):
        clean_both += 1
    sets = []
    for x in reps:
        d = {}
        for fd in x["findings"]:
            k = (fd["type"], fd["region"])
            if k not in d or fd["severity"] > d[k]["severity"]:
                d[k] = fd
        sets.append(d)
    for k in set(sets[0]) & set(sets[1]):
        a, b = sets[0][k], sets[1][k]
        sev = (a["severity"] + b["severity"]) / 2
        frames = sorted(set(a["frames"]) | set(b["frames"]))
        cats[k].append((f, index.get(f), sev, frames, a["description"][:140]))
print(f"{name}: strips reviewed twice {reviewed}, clean per both reviewers {clean_both}")
rank = sorted(cats.items(), key=lambda kv: -sum(x[2] for x in kv[1]))
for (typ, reg), items in rank:
    sev = sum(x[2] for x in items) / len(items)
    hi = sum(x[2] >= 3 for x in items)
    print(f"\n{typ} / {reg}: {len(items)} strips (severity>=3: {hi}), mean severity {sev:.1f}")
    for f, s, sv, fr, desc in sorted(items, key=lambda x: -x[2])[:6]:
        print(f"   {f} start {s} sev {sv:.1f} frames {fr[:8]} | {desc}")
