# Continuing the Teeth Filter work

This is where the project stands, how to use it, and how to keep improving it, with or without Claude.

## 1. Using it

`run.bat` is unchanged. The default method is now **anchor** (v10). It needs two files in the project folder, both in the repo:

- `teeth_model.npz`: your dental geometry (where the missing teeth sit relative to your skull).
- `teeth_seg.onnx`: a small mouth segmenter trained on your face. It runs on the GPU through DirectML, in about 4 ms.

If either file is missing, the app falls back to the old method with a message. You can also force a method: `run.bat --method edge` (v9) or `--method legacy` (v5).

Keys that matter for the anchor method:

| Key | Action |
|---|---|
| `f` | filter on/off (before/after check) |
| `u` | upper-arch fill on/off |
| `l` | lower-arch fill on/off |
| `d` | debug: red tint shows the replaced pixels |
| `r` | record raw + filtered clips into `recordings/` |
| `c` | next camera (resets tracking) |

The first ~3 frames after your teeth become visible are left unfilled while tracking locks on. The lock position is saved to `settings.json` on quit (`anchor_offset`), so the next session starts locked.

Test a recording offline: `run.bat --video recordings\<name>_raw.mp4 --out test.mp4`

## 2. How it works (short)

1. MediaPipe landmarks go through a rigid fit on skull-stable points (nose bridge area), which gives a **dental frame**: a 320x260 crop in which your upper teeth never move.
2. An edge tracker finds the gap-side edge of your present upper central incisor in that crop and corrects MediaPipe's pose error (usually 3-5 px). The mirror axis sits just left of that edge.
3. The segmenter gives the mouth opening, so lips always stay in front.
4. **Upper fill:** each column in the missing slots (central + lateral) gets the live pixels from the mirrored column on the other side of the axis. It covers from just under your lip down to the mirrored tooth's own incisal edge.
5. **Lower fill:** your healthy lower teeth (viewer's left) are mirrored onto the decayed ones (viewer's right), underneath the upper teeth.
6. The result is warped back into the frame. There is no gap detection, so there's nothing to flicker.

Everything is in `dental.py`. Its tunables are attributes on `AnchorFill`:

- `extend_px`: how far above the donor's visible tooth the fill may extend (default 6).
- `bottom_margin`: overlap below the incisal edge (default 1).
- `k_new`: tracking smoothing (0.7; higher follows faster).
- `upper` / `lower`: which arches are filled.
- The slot width is the `66.0` in `__init__`: 66 = central + lateral, 88 would include the canine.

## 3. Where quality stands

Blind A/B judging used 4 AI judges on the same 40 frames from 6 clips; `lab/make_judge_set.py` builds the images.

| Version | Win rate vs raw | Naturalness (1-10) | Gap visible |
|---|---|---|---|
| raw video | – | 4.1-4.5 | 73-77% |
| v9 (old edge method) | – | 4.3-4.7 | 53-62% |
| v11 (anchor, upper only) | 75% | 6.0 | 8% |
| v13 (anchor, upper + lower, current) | 75% | 5.9 | 25%* |

\*v13 also counts lower-teeth gaps. Head to head on frames where the lower fill is active, lower on beat lower off 89% of the time (naturalness 5.6 vs 4.0).

Speed: ~28.5 fps with a 30 fps camera. Face tracking runs on its own thread (`Capture` in `teeth_filter.py`). MediaPipe takes ~22 ms, the fill ~17 ms, and they overlap.

## 4. Known problems, in priority order

1. **Drawn-looking dark outline along the incisal edges**, a symmetric "gull-wing" shape. The mirrored tooth brings its shadow line with it. Try soft-fading the bottom 1-2 px of the fill, or darkening less.
2. **Midline line runs straight through both arches.** The upper and lower mirrors use the same axis. Offset the lower axis by 1-2 px, or soften the lower seam.
3. **Too-perfect symmetry** of the two centrals. A slight brightness or shape jitter on the mirrored copy would help.
4. **Wide smiles:** the mirrored lateral can look grey or flat next to the bright lower teeth that show where your upper canine is missing.
5. **Lower fill** occasionally leaves a bright, flat "slab" at the lower lip line (`lower_fill` in `dental.py`).
6. **Short crescents:** when only a sliver of the donor tooth is visible, the fill is a thin bright crescent.
7. **Tracking re-lock pops** after the mouth opens. Hidden-frame error is ~2.3 px (`lab/align_eval.py`).
8. Not yet tested **live on your webcam and in OBS**. Only recordings were used.

## 5. Recording better data (biggest easy win)

The research pass recommended one 60-90 s calibration clip. Record it with `r`, same webcam, auto-exposure and white balance locked, even front light:

- neutral, then a broad smile held for 3 s, then "eee", then normal talking;
- slow head turns (about ±25°) and nods while smiling;
- teeth together, then lips relaxed and apart;
- a pass sitting ~30% closer.

More varied clips improve the segmenter and the tooth geometry. Retrain after recording (section 6).

## 6. Rebuilding the model from recordings

Run from `lab/` with the venv (`..\.venv\Scripts\python.exe <script>`). The cache takes ~6 GB (gitignored). About 1.5 h total on the RTX 3080.

1. `bootstrap_cache.py`: extracts the skull reference shape from `teeth_model.npz` (only needed on a fresh clone).
2. `cache_landmarks.py`: MediaPipe landmarks for every `recordings/*_raw.mp4`. It also reads four OBS clips from `%USERPROFILE%\Videos` if present; those aren't in the repo.
3. `teacher_labels.py`: runs the SegFormer + BiSeNet face parsers (downloaded from Hugging Face on first use, ~140 MB) on every 2nd frame. About 15 min.
4. `train_seg.py --out seg_w16`: trains the mouth segmenter. About 40 min. Don't run another GPU job at the same time: 10 GB fills up and everything slows 10x.
5. `export_seg.py seg_w16`: exports to ONNX.
6. `build_atlas3.py` then `make_model_c.py`: rebuild the dental geometry.
7. `package_model.py seg_w16`: writes `teeth_model.npz` + `teeth_seg.onnx` to the project root.

Checking a change:

- `run_dental.py <tag> <clip keys...>`: renders clips through `dental.py` into `out/<tag>/`.
- `dbg_dental.py <clip> <frames> out.jpg`: crop, fill and alpha for specific frames.
- `align_eval.py`: tracking metrics.
- `make_judge_set.py <tag>`: blind A/B images (it also compares against `out/v9/` renders if they exist).

Clip keys look like `2026-09-23_05-46-00_raw`.

## 7. Continuing with Claude Code

`CLAUDE.md` has the technical notes a new session needs. Useful opening prompts:

- "Read HANDOFF.md and CLAUDE.md, then fix known problem #1 and show me before/after frames from lab/dbg_dental.py."
- "Rebuild the model with my new recordings (HANDOFF.md section 6) and compare against the current one."
- "Run the blind judge comparison for the current version against v13."

**Local sessions** (this PC) can do everything. **Cloud sessions** get a clone of the GitHub repo, recordings included, but likely no GPU and no Windows. They're fine for code changes, docs and CPU analysis. Rendering a clip on CPU works but is slow. Retraining and anything live (webcam, OBS, DirectML) needs this PC.

## 8. Without Claude

- Tune by eye: change a tunable in `dental.py` (section 2), then run `run.bat --video recordings\<clip>_raw.mp4 --out test.mp4` and watch it.
- Before an interview: open OBS → Browser Source `http://127.0.0.1:8765/`, then press `f` a few times while smiling and talking to check. Even lighting from the front makes the biggest difference.
