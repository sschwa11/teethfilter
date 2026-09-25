import collections
import json
import sys

res_file, name, new = sys.argv[1], sys.argv[2], sys.argv[3]
t = open(res_file, encoding="utf-8").read()
res = json.loads(t[t.find("{"):])["result"] if not t.lstrip().startswith("[") else json.loads(t)
key = {k["file"]: k for k in json.load(open(f"out/judge_{name}/key.json"))}
cnt = collections.Counter()
nat = collections.Counter()
gap = collections.defaultdict(list)
fl = collections.defaultdict(list)
per_clip = collections.defaultdict(lambda: collections.Counter())
notes = []
for judge in res:
    for r in judge["ratings"]:
        k = key[r["file"]]
        sn = "A" if k["A"] == new else "B"
        so = "B" if sn == "A" else "A"
        v = r["less_lower_flicker"]
        w = "new" if v == sn else ("old" if v == so else "tie")
        cnt[w] += 1
        per_clip[k["clip"][11:19]][w] += 1
        m = r["more_natural"]
        nat["new" if m == sn else ("old" if m == so else "tie")] += 1
        gap["new"].append(r["lower_gap_frames_" + sn])
        gap["old"].append(r["lower_gap_frames_" + so])
        fl["new"].append(r["flicker_" + sn])
        fl["old"].append(r["flicker_" + so])
        notes.append((r["file"], k["clip"][11:19], k["start"], sn, r["notes"]))
n = sum(cnt.values())
mean = lambda v: sum(v) / len(v)
print(f"{name}: n={n} less lower flicker new/old/tie {cnt['new']}/{cnt['old']}/{cnt['tie']}  more natural {nat['new']}/{nat['old']}/{nat['tie']}")
print(f"  lower-gap frames (of 8) new {mean(gap['new']):.2f} vs old {mean(gap['old']):.2f}   flicker new {mean(fl['new']):.2f} vs old {mean(fl['old']):.2f}")
for c, v in per_clip.items():
    print(f"  {c}: new/old/tie {v['new']}/{v['old']}/{v['tie']}")
if len(sys.argv) > 4:
    for f, c, s, sn, nt in notes:
        print(f, c, s, "new=" + sn, "|", nt[:260])
