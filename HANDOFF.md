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
4. **Upper fill:** a *stable tooth model*. The mirrored image of your present central and lateral is accumulated over time: texture, silhouette, and a brightness gain matched to the current frame. The fill is drawn from that model instead of fresh pixels each frame, which is what removed most of the flicker. The lip matte decides what is visible. The model is saved to `teeth_memory.npz` on quit, so a new session starts with it.
5. **Lower fill:** your healthy lower teeth (viewer's left) are mirrored onto the decayed ones (viewer's right), underneath the upper teeth.
6. The seam where the fill meets your real tooth is healed on both arches, so it reads as a thin interproximal line instead of a dark crack.
7. The fill appears instantly when the mouth opens and fades out over about 3 frames when it closes. Frame-to-frame image registration keeps the axis locked between tracker measurements.
8. The result is warped back into the frame.

Everything is in `dental.py`. Its tunables are attributes on `AnchorFill`:

- `extend_px`: how far above the donor's visible tooth the fill may extend (default 6).
- `bottom_margin`: overlap below the incisal edge (default 1).
- `k_new`: tracking smoothing (0.7; higher follows faster).
- `st_rate`: how fast the stable tooth texture adapts (0.12).
- `gain_lo`/`gain_hi`/`gain_k`: brightness-matching limits and speed (0.4 / 1.8 / 0.7).
- `rise_u`: how fast the upper fill appears (1.0 = instantly).
- `seam_band`: seam healing on/off.
- `slot_w`: how far the upper fill reaches from the midline (88 px = central, lateral and canine).
- `l_span`: how far the lower fill reaches from the lower midline (100 px).
- `arch_on`, `arch_a`: head-turn correction on/off and the curvature of the jaw it assumes (0.0055).
- `robust`, `max_step`, `use_reg`: tracking (median gating on, 1.5 px/frame cap, registration off).
- `live_ext`: how far below the model's tooth edge the fill may follow your real tooth (6 px).
- `stable = False` goes back to the per-frame live mirror (v13 behaviour).
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

Flicker (v20-v23) was judged by 4 blind AI judges on 13 pairs of 8-consecutive-frame strips against your live-recorded v13 output (`lab/make_seq_judge.py`, `lab/score_seq.py`):

| Version | Less flicker (new / v13 / tie) | Mean flicker 0-10 (new vs v13) |
|---|---|---|
| v20 stable tooth + registration | 17 / 12 / 23 | 2.62 vs 2.85 |
| v22 + correct brightness gain, instant appear | 12 / 4 / 36 | 3.17 vs 3.35 |
| v23 (current) + seam healing | 11 / 10 / 31 | 3.15 vs 3.06 |

v24 (tracking): on your 2026-09-25 recording, the fill sometimes sat 4-8 px beside your real tooth and left a dark strip. Frame-to-frame registration was causing drift, and good edge measurements were being rejected. Registration is now off, measurements are gated against the median of recent ones, and the axis moves at most 1.5 px per frame. The fill also extends down to your real tooth's current biting edge. Tracking error dropped from 1.94 to 1.31 px, and frames more than 3 px off dropped from 22% to 9% (`lab/track_score.py`). On the 8 frames where v23 and v24 differ most, v24 looked better in 5 and worse in 1 (by eye, not blind-judged).

v25 (lower fill): on your 01-10-45 recording, most of the gap you still saw while talking was the lower teeth (viewer's right), not the upper. The lower fill waited 2 frames and then ramped in at 60%, so short mouth openings showed the gap. It now appears immediately (`l_need=1`, `rise_l=1.0`). In a thin sliver it can look slightly too white.

v26 (lower axis): your lower jaw shifts sideways relative to your skull as you talk, so your healthy lower teeth end anywhere from about 4 px left to 11 px right of the upper midline. The lower fill used the upper midline as its mirror, so it either overlapped your real teeth or left a dark gap next to them. It now finds the edge of your healthy lower teeth every frame and mirrors about that edge. The brightness threshold comes from the lower teeth themselves (they're dimmer than the upper), and gaps between mirrored teeth are bridged up to 15 px wide. Remaining: a dark notch at the lower center line in some frames.

v27: with the mouth wide open, the lower axis sometimes locked onto the stub teeth on the far side of the gap. It now takes the first real dark run coming from the healthy side, which fills the lower gap in wide-open and laughing frames. Known: a one-frame tooth patch can appear on a finger when a hand sweeps over the mouth; "O"/pucker shapes still show the upper gap; with the mouth wide open the fill takes ~26 ms per frame.

v28: occlusion guard. The fill only replaces pixels darker than itself, so a hand or finger in front of the mouth is no longer painted over. It also trims the too-white lower slivers slightly. The lower fill now works only in its own region, which cuts about 12% of CPU per frame. Still known: "O"/pucker shapes. Your face landmarks shift when you pucker, so the fill lands about 5 px off. The tracker's correct measurements in those frames are weak, and accepting weak measurements made tracking worse overall (tested with `lab/track_score.py`).

v29-v34 (lower-teeth flicker): the lower fill was switching on and off. The columns next to your healthy lower teeth were dropped in some frames and kept in others, so a dark strip flashed at the lower midline. Fixes:
- The fill remembers each column for 6 frames.
- Short gaps back to the mirror line are extended.
- Bridged columns take texture from the nearest solid tooth instead of the dark gap.
- The mirror line search reaches further left (your jaw shifts further when the mouth is wide open) and uses a threshold from the lower teeth themselves.
- The lower teeth may end up to 12 px above the lip line (the lip curves down near the middle).
- The fill is 76 px wide.

Measured on your last three recordings (`lab/lower_blink.py`), frames where the lower gap still shows dropped from 25% / 16% / 17% (v28) to 9% / 6% / 14%, with about half as many frame-to-frame jumps. Blind AI judges (6 judges, 22 lower-teeth strips, v33 vs v28) preferred the new lower fill 65 to 25, with 42 ties. They counted a visible lower gap in 1.5 of 8 frames versus 2.7. Remaining: with the tongue raised in a wide-open mouth, a bright block can appear above the lower teeth; flat-looking blocks sometimes appear at the right end of the lower fill.

v35: on your 10-05-06 recording (live v34), frames where the lower gap still shows were 5.4% (v28 on the same clip: 9.2%). A blind defect review found the most common remaining flaw was a thin too-white crescent along the bottom of the lower fill where it meets the lip. The bottom 4 px of the lower fill are now shaded up to 25% (`l_lip_shade`). Still seen: a one-frame dark notch below the filled lateral when the mouth is barely open (it mirrors the real dark space on the other side), and faint/streaky upper fill in nearly closed slits.

v36: the lower fill fades out over 6 px where the mirrored teeth end, instead of stopping in a hard, squared-off edge (`l_feather`). Where the opening it shows through is much thinner than the teeth, it is shaded up to 30% darker (`l_slit`), so thin slivers no longer look like bright shards. Gap metric on four recordings: 9.3% of frames vs 9.0% for v35, i.e. neutral. A left/right lighting match (`l_light`) was tried and rejected: your side lighting changes between sessions and the estimate was too noisy (gap 11.7%, more jitter).

v37 (upper teeth flicker, gap between the front teeth):
- **Tongue:** the mouth-outline model treats your tongue as "not mouth", so with your tongue up the detected opening split in two. The cleanup kept only the biggest piece, and when that was the lower one, the upper fill vanished for a frame. It now keeps every piece inside your lip outline.
- **Gap between the front teeth:** when the mouth is barely open, the tracker can go 40+ frames without a confident reading and drift 3–6 px. Because the fill is a mirror image, that shows up as a dark gap twice as wide between the filled tooth and your real one. Each frame, the filter now finds where your real tooth actually starts and stretches the fill to meet it (`seam_max`).
- **Result** (`lab/seam_gap.py`): frames with a gap of 3 px or more fell from 4.7% to 1.9% (26 Sep recording) and from 6.3% to 4.1% (25 Sep). Single-frame upper dropouts with the tongue up are gone. Lower teeth unchanged.

v38 (tongue flicker): with your tongue raised and touching the backs of your upper teeth, the mouth-outline model sometimes marks only fragments of the opening near the teeth, and some frames it marks none of it. The upper fill then blinked on and off, or was cut into a notch where the tongue was. Two fixes:
- Tongue-shaped holes in the mouth outline no longer cut the upper fill. The fill covers everything between the top and the bottom of the opening in each column.
- When the band at your upper teeth suddenly nearly disappears while your lip outline stays about the same and the rest of your mouth is still open, the filter keeps the previous frame's opening there for up to 10 frames (`drop_hold`). The "rest still open" check matters: without it the filter painted a tooth sliver onto your lip for a frame when your mouth was closing, because the mouth-outline model sees closing lips a frame before the face landmarks do.

On your tongue recording (00-24-05), complete one-frame dropouts of the upper fill fell from 17 to 3. Frame-to-frame changes in the fill's shape there dropped 28%. The other recordings are unchanged, since the hold only kicks in with the tongue up. Seams and the lower teeth are also unchanged.

v39 (retrained mouth-outline model): the small model that finds the mouth opening had only been trained on recordings up to 23 Sep. Those had almost no tongue or head-down footage, so it labelled a raised tongue as lip, which caused the v37/v38 tongue flicker. It was retrained on all 27 usable recordings (`teeth_seg.onnx`). Two clips it never saw were held back for testing:
- On the tongue clip (00-24-05), agreement with the reference labels rose from 69% to 96%. Complete upper-fill dropouts went from 3 to 0.
- On the head-down clip (01-10-00), agreement rose from 55% to 93%, and the error at the upper lip edge fell from 12 px to 4 px.
- The lower-teeth gap and the front-teeth seam are unchanged or slightly better on four other recordings.

Still known: with your head tilted well down and your mouth wide open, the upper teeth show only as a thin strip under the lip. Even the large reference models label that strip as lip, so the filled tooth can still be missing there. A detector that mirrors your visible real tooth into the gap in that case (`band_rescue`) exists but is off: it only triggered in some frames, so it made a small patch blink. Keeping the camera at eye level avoids the problem.

v40 (wider fills with head-turn correction):
- **Wider:** the upper fill now also covers the canine gap on the viewer's left (`slot_w` 88, was 66). The lower fill reaches about one tooth further toward the corner (`l_span` 100, was 76).
- **Head-turn correction:** your side teeth sit further back along the curve of your jaw, so when you turn your head they shift relative to the front teeth. A flat mirror copy then puts them 5–13 px off at the canine. The filter now reads your head angle every frame and corrects for the curve (`arch_on`, `arch_a`).
- **Measured** on four recordings, as the share of dark spots that your healthy side shows should be tooth: at the upper canine it fell from 91–92% to 9–25%; beyond the old lower fill, from 72–95% to 25–47%; in the existing lower area it also improved a little. The front seam, the lower gap near the middle, and upper dropouts are unchanged. Nothing is painted onto your lips.
- **Cost:** about 3–4 ms more per frame, roughly 14%. Live speed is unchanged (see v41).
- **Limit:** the mirror can only copy teeth your healthy side shows in that same frame. When you turn so the damaged side faces the camera, the matching healthy teeth are hidden behind your cheek, and that part stays unfilled.

v41 (front tooth flicker): the filled front tooth was drawn from a remembered image of your tooth. That memory had become dark and streaky: it was saved darker than your real tooth and brightened back up at the limit (x1.8), so the fill didn't track your real tooth. It showed as a grey block when your lips were nearly closed and your real tooth was in shadow, and as streaks with your teeth together. Now, wherever your real front tooth is visible, the fill copies it directly (mirrored) that frame, and the memory is only used where your real tooth is hidden, e.g. behind your lip (`live_first`). The copy fades out within 7 px of the midline, so the seam stays as before (`lf_seam`). The memory is also saved at its true brightness now.
- Measured on your two new clips (`lab/pair_flicker.py`), sudden brightness jumps of the fill relative to your real tooth fell 8% on the long clip and 46% on the short one. The fill's glow in a dark slit halved. On three older recordings the jumps fell 11-27%. The front seam is unchanged.
- Still visible: a thin bright bar for one frame when your mouth first opens after an "O" shape, and the tooth-edge tracking can't get a reading in those frames.
- Tried and left off: stronger up-down tracking (`y_track`) and fitting the fill's height to your real tooth each frame (`vfit`); neither measurably helped.
- Tested live on 28 Sep (recording 14-26-26): the live output matches the offline replay, and brightness jumps were about 25% fewer than v40 would give on the same clip. Speed was 17.5 fps while recording, the usual 17–21 fps (recording saves two videos at once).

v42 (grey patch when your lips are nearly closed): when your lower lip rises over your teeth, only a dark slit is left where the missing teeth are, and your real tooth next to it is dim. The fill stopped at the edge of the detected opening, so the slit's soft dark edge stayed around it: for 1–3 frames the fill looked like a flat grey patch with a dark outline. Now, in those nearly closed frames, the fill's edge blends into your lips according to how dark each edge pixel is, the way your real tooth meets your lips (`soft_edge`). It never makes a spot brighter than the matching spot on your real side or the nearby lip, and open-mouth frames are unchanged (there the dark band under your upper lip is the lip's own shadow).
- Measured (`lab/pair_flicker.py`): on the new recording, frames where the fill pops in or out next to your real tooth fell from 16 to 9. On three older recordings: 4 to 2, 4 to 3, 3 to 3. Brightness jumps, the front seam and the lower teeth are unchanged (within ±3%). Cost: about 1 ms per frame on average.
- Tried and left off: filling those frames with a straight mirror of your real side (it smears your lip's bright highlight into the gap), and filling the dark wedge under a slanted fill (it could only half-fill it, leaving notches).
- Still visible: in those frames the fill is still a little greyer than your real tooth, which picks up pink from your lips, and when the fill's bottom edge is slanted a dark wedge can remain under it.

v43 (rounded "oh" mouth): with your lips pushed forward in an "oh", your gum and the stubs of the missing teeth show under the upper lip, lit to a medium grey-pink. The mouth-outline model (and the large reference models it learned from) is only about half sure that band is mouth, so the fill stopped below it: a thin sliver, a "U" shape or a small blob next to your real tooth, with the gap still showing. Now, when your mouth is narrow and tall (the pucker shape), the fill extends up through that uncertain band to the lip line wherever it's clearly darker than your real tooth (`ramp_ext`). It only switches on for that shape; on four older recordings it changes 2 frames in total.
- Measured on the new recording (`lab/mirror_cover.py`: how much of your mirrored real tooth the fill covers): in the "oh" stretch it rose from 66% to 83%, and frames less than half covered fell from 16 to 3. Flicker and the front seam are unchanged.
- Still visible: over the stubs the fill is a smooth band rather than tooth-shaped, and occasionally there's a small spike or bump at its edge.

At thumbnail size the judges mostly call ties. At full size, v23 beat v22 for naturalness 10 vs 1 (19 ties), because the dark midline crack is gone.

\*v13 also counts lower-teeth gaps. Head to head on frames where the lower fill is active, lower on beat lower off 89% of the time (naturalness 5.6 vs 4.0).

Speed: ~28.5 fps with a 30 fps camera. Face tracking runs on its own thread (`Capture` in `teeth_filter.py`). MediaPipe takes ~22 ms, the fill ~17 ms, and they overlap.

## 4. Known problems, in priority order

0. **Brief dark notch when the mouth first opens**, for 1 frame. This is the most common remaining flicker complaint. It happens while the tracker and lip matte catch up.
1. **Drawn-looking dark outline along the incisal edges**, a symmetric "gull-wing" shape. The mirrored tooth brings its shadow line with it. Try soft-fading the bottom 1-2 px of the fill, or darkening less.
2. The top edge of the fill against the lip can look slightly stepped.
3. **Too-perfect symmetry** of the two centrals. A slight brightness or shape jitter on the mirrored copy would help.
4. **Wide smiles:** the mirrored lateral can look grey or flat next to the bright lower teeth that show where your upper canine is missing.
5. **Lower fill** occasionally leaves a bright, flat "slab" at the lower lip line (`lower_fill` in `dental.py`).
6. **Short crescents:** when only a sliver of the donor tooth is visible, the fill is a thin bright crescent.
7. **Tracking re-lock pops** after the mouth opens. Hidden-frame error is ~2.3 px (`lab/align_eval.py`).
8. **Recording costs speed:** with `r` on, the filter runs at 17–21 fps because it saves two videos at once.

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
4. `train_seg.py --out seg_w16`: trains the mouth segmenter. About 40-80 min. The clips in `VAL_KEYS` are held out for testing; `eval_seg.py <keys> <model names>` compares models per clip. Don't run another GPU job at the same time: 10 GB fills up and everything slows 10x.
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
