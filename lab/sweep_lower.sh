#!/bin/bash
tag=$1; shift
for K in 2026-09-25_08-03-41_raw 2026-09-25_02-36-08_raw 2026-09-25_01-18-25_raw; do
  ../.venv/Scripts/python.exe run_dental.py $tag $K "$@" > /dev/null 2>&1
done
for K in 2026-09-25_08-03-41 2026-09-25_02-36-08 2026-09-25_01-18-25; do
  ../.venv/Scripts/python.exe lower_blink.py ${K}_raw ../recordings/${K}_raw.mp4 out/$tag/${K}_raw_full.mp4 2>/dev/null | tail -1
done | ../.venv/Scripts/python.exe -c "
import sys,json
rs=[json.loads(l) for l in sys.stdin]
g=sum(r['gap_left'] for r in rs)/3; b=sum(r['gap>0.3'] for r in rs)/3; j=sum(r['jitter'] for r in rs)/3; n=sum(r['jumps>0.3'] for r in rs)
print('$tag', ' '.join('$*'.split()), '| per clip gap>0.3', [r['gap>0.3'] for r in rs], '| MEAN gap_left %.3f gap>0.3 %.3f jitter %.3f jumps %d' % (g,b,j,n))"
