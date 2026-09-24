# Teeth Filter

Real-time webcam filter that fills missing-tooth gaps so they don't show on video calls. Output goes to OBS (Browser Source at http://127.0.0.1:8765/, or Window Capture of the "Teeth Filter" window), then OBS Virtual Camera goes to Zoom/Teams/Meet. Windows 11, Python venv in `.venv`, launched with `run.bat`.

## Conventions
- No comments in code.
- Keep responses succinct; don't restate the question.

## Pipeline (teeth_filter.py)
1. Capture webcam (DirectShow, MJPG, 1280x720@30). MediaPipe FaceLandmarker (VIDEO mode), landmarks smoothed with adaptive EMA.
2. `mouth_frame` builds an affine from the inner mouth corners (78, 308) into a canonical 320x260 crop with the mouth 200px wide. `lip_curves` and `mouth_mask` come from the inner-lip landmarks. NOTE: on Scott's webcam the inner-lip landmarks sit several px off the true lip edge, so nothing may rely on lip-line accuracy.
3. Tooth vs dark: Otsu split of inner-mouth L (EMA), falling back to `dark * max(p90, skin L)`.
4. `find_gaps`: horizontal morphological close of the bright-teeth mask (kernel = max_gap * 200px) minus bright = dark holes with teeth on both sides in the same row. Components are filtered by min width (min_gap), height, h >= 0.25w, and nearness to a lip (within 0.1 MW); the upper/lower arch is assigned by position. The learned prior (auto-calibration) rejects components where prior < 0.2 once ready.
5. `update_tracks`: match gaps across frames by column IoU; confirm after 2 hits, keep for 10 frames after the last hit.
5b. `find_mirror` (upper arch): per column, count lip-free bright pixels in the upper half of the opening; columns with < 0.018 MW bright whose mirror column has > 0.045 MW become mirror fills. The mask is the donor tooth's vertical span, shifted by the lip offset. `st.mcols` (EMA) remembers missing columns so fills persist while talking. This handles Scott's main gap (upper front tooth just left of center), which is in shadow when he talks, so there are no teeth on both sides.
5d. Mirror mask = interpolated, median-smoothed donor top/bottom span across the segment (no holes). Candidate columns are closed horizontally (1x11) so the donor's interproximal lines don't split a fill into stripes. Tracks that overlap on the same arch are merged, and mirror tracks win. There is no stale-pixel reuse between frames: held tracks re-render from the current frame (the old hold logic caused ring artifacts). A bridge pass inpaints narrow dark slivers (under 13px) beside mirror fills.
5c. `lip_free_bright`: bright = L >= Otsu and a* below the midpoint of lower-lip a* and tooth a* (his lips ~150 vs teeth ~140-145).
6. `fill_gap` (source "self"): clone the donor columns (mirror across the midline first, then the neighboring tooth on the left/right), vertically aligned by lip offset, with an L gain matched to the adjacent real teeth. Fallback: horizontal interpolation between neighbor columns. Sources "reference"/"procedural" use atlases via `fill_gap_atlas`.
7. Alpha = dilated gap mask, feathered, temporal EMA, faded by mouth openness (min_open 0.05); inverse-warped into the mouth ROI only.
8. MJPEG server streams the clean output. Preview: HUD (h), debug (d). `r` records raw+filtered mp4s into recordings/. `--video file` runs offline for testing.

## v6 default method: edge mirror (`--method edge`; `--method legacy` = v5 pipeline)
`find_edge_mirror`: connected components of lip-free bright teeth in the upper/lower 65% of the opening. For each tooth blob that hangs from its lip, walk sideways from its edge through dark columns (middle rows of the tooth). If the dark run is >= 0.45 * tooth width, flip the blob (vertically column-filled) across that edge (`axis`) into the gap, up to one tooth width (<= 0.17 MW). Rules: fill center |u| <= 0.4; unbounded gaps (no tooth on the far side) are allowed only for the upper arch with the edge |u| < 0.32; bounded gaps must be <= 2.2 tooth widths; lower fills need height >= 0.05 MW. `fill_edge` remaps the crop with x' = 2*axis - x. Tracks carry `axis`. Anatomical basis: Scott's missing upper-left central incisor is the mirror image of the present central incisor across their shared edge.
Tuned on "2026-09-22 23-02-18.mp4" (OBS recording, 1080p, bright light, whole face, mouth ~175px at 720p): the best data so far.

v7 additions (tuned on recordings/2026-09-22_23-36-22 and 23-36-51, bright/overexposed pink light):
- `inside_up` = mouth mask extended 0.05 MW upward, because landmarks put the upper lip line below the real incisor edge. Above yu, pixels count as teeth only if a* < tooth a* + 3 (to reject lip highlights). `valid_donor` drops donor columns whose tooth doesn't reach below yu (+2).
- Upper tooth region is at least 0.12 MW tall, so teeth-together frames don't truncate the donor.
- `st.edge_mem` remembers the central upper gap (u of the axis, side, tooth width, confidence). Once confident (>= 12), `edge_from_memory` re-finds the tooth edge near that axis and fills even when the gap isn't dark (lower teeth showing through).
- Known remaining: teeth-together smiles early in a session (memory not yet built) and small slivers near the right lateral.

## AI fill (v8)
- `ai_model.py`: TeethNet, a 3.7M-param U-Net (4ch in = masked BGR + mask, 3ch sigmoid out, bilinear upsampling), input 120x240 = canonical crop AI_CROP (70:190, 40:280).
- `train_ai.py`: builds `ai_data/dataset.npz` from recordings/*_raw.mp4 (canonical crops, tooth-blob label maps, inside_up, real-gap masks from find_edge_mirror). Self-supervised: masks a tooth-width window of a real tooth blob (column-filled), then learns to repaint it. Loss = 6*hole L1 + 0.5*L1 + 2*gradient + 0.08*VGG16 perceptual. Adam 2e-4 with cosine decay, 8000 iters, batch 16. Previews go in ai_preview/ (lower rows = real gaps). Exports teeth_ai.onnx (opset 17, legacy exporter).
- `train_ai.bat` installs torch cu128 into .venv if CUDA isn't available, plus onnx.
- The filter loads teeth_ai.onnx via onnxruntime-directml (DmlExecutionProvider) when present; `--fill auto|ai|mirror`, `m` toggles. process(): the union of gap masks (dilated 5x5), clipped to AI_CROP, goes to AIFill.run; the output replaces the mirror fill.
- Verified in the sandbox on CPU only (400 iters: pipeline OK, fills still flat). Real training happens on Scott's RTX 3080 laptop GPU (8 GB).

## AI fill v2 + anti-flicker (v9)
- The v1 model (plain inpainting, 4ch) painted darkness into real gaps, because the context around them is dark mouth. v2 is a hint-conditioned refiner: 7ch in = masked image, mask, hint (the mirror fill). In training the hint is a mirrored neighbor strip of the same tooth blob (70%) or a shifted/color-jittered ground truth, plus a color/blur jitter. Dataset v2 (`ai_data/dataset_v2.npz`) keeps only blobs >= 50% inside the true mouth mask and open_ratio >= 0.07 (drops lip false positives), and stores real-gap mirror hints for previews. The old model was renamed to teeth_ai_old_v1.onnx.
- AIFill detects the input channels from the ONNX input shape (4 = old, 7 = new) and gets hint = the fill_edge composite.
- Anti-flicker: a central-gap tracker (`st.cax` smoothed axis, `c_mask` EMA, holds 8 frames with decay) driven by edge_mem (conf >= 6); falls back to a raw detection near the axis with the same side. Raw detections near the central axis are suppressed. Other edge gaps need >= 3 hits. Edge tracks are never merged (the merge bug mixed axes). Alpha rises at 0.5 and falls at 0.3 per frame, and is masked by inside_up after the EMA. Fill pixels blend with the previous rep (k = 0.5, or 0.2 when moving). Track ttl is 15. edge_memory is saved in settings.json and loaded with conf 8, so the tracker locks on immediately.
- Flicker metric (mean |delta(out - raw)| over the filled region): r7 23.6 -> 17.4, raw5 19.3 -> 15.6, with more frames filled.

## AI fill v3 (residual harmonizer)
The v2 model (hint as input) learned to ignore the hint. In real gaps it restored the dark gap, because self-supervised training only teaches "reconstruct what was there". v3 constrains the output: out = clamp(base + 0.25*tanh(net)), where base = the hint inside the mask and the image outside it. The model can only adjust color, lighting and edges; it can't erase the mirrored tooth. v2 was renamed to teeth_ai_old_v2.onnx.

## History / lessons
- v1 filled every dark pixel in lip-anchored bands with procedural teeth, which produced a tiled grid along both lips, painted the natural space between the upper and lower teeth, and drew artifacts when the mouth was nearly closed. Auto-calibration then shrank the tooth length (~0.4) because lip landmarks were off. Replaced by the gap detection + self-cloning above.
- Scott's first test video (2026-09-22) was recorded through v1, so its frames contain v1 artifacts. Ask for raw clips (the `r` key) before tuning further.

## Auto-calibration (`a` key / `--auto`)
On stable frames: learns a per-column missing-tooth probability for each arch (settings.json missing_upper/missing_lower, 320 values) and nudges extent toward the reach of the real teeth. After 150 frames the prior is "ready" and suppresses detections in columns with prior < 0.2. `x` clears it.

## Camera selection
`list_cameras` uses pygrabber (DirectShow names; index order matches cv2.CAP_DSHOW), falling back to probing indices 0-9. OBS Virtual Camera is listed (labeled "OBS output"; Scott uses OBS to crop/position his webcam), with a loop warning when selected. OBS has only one virtual camera, so if it feeds the filter, the filtered output can't also go out through it. The choice is saved to `last_camera.txt`. Flags: `--pick`, `--list-cameras`, `--camera N`, `--camera-name substr`; `c` cycles cameras live.

## Other flags / keys
`--ref img`, `--max-gap 0.24`, `--min-gap 0.025`, `--extent 0.7`, `--mirror-extent 0.55`, `--min-open 0.05`, `--dark 0.55` (fallback only), `--auto`, `--no-upper`, `--no-lower`, `--window-width 852`, `--window-height 480` (preview only; capture stays 1280x720), `--image in.jpg`, `--video in.mp4`, `--out path`, `--port 8765`.
Keys: f toggle, d debug, h HUD, r record, a auto-calibrate, x clear learned gaps, u/l arches, p fill source (self/reference/procedural), [ ] max gap, - = min gap, , . extent, c camera, q quit.

## Status
- The live path works on this machine (mediapipe 1.0.1 / opencv 5.0 venv); camera picker works.
- The v2 gap-fill was only validated offline on Scott's v1-contaminated test video (runs ~6 ms/frame in the sandbox).
- v3 tuned on recordings/2026-09-22_18-56-41_raw.mp4 (mouth only ~100px wide at 720p). Good: the front gap while talking or smiling, and the smile gaps. Still weak: the narrow dark notch at center in some talking frames, and wide-open speech frames where the whole left side is dark.
- v4 tuned on recordings/2026-09-22_20-11-23_raw.mp4 (fed through ManyCam: watermark and pillarbox). Remaining weak spots: the self-mirrored sliver at the exact center, and wide-open vowels.
- v5 tuned on recordings/2026-09-22_21-44-57_raw.mp4 (sitting closer: mouth ~115px; still ManyCam). Mirror donors must "hang" from the top of the opening (first bright <= yu + 0.4*open), so lower teeth aren't copied into the upper row. Memory-driven mirror fills require the target to be nearly empty (n < 1.5*lo_t). Tried and reverted: dropping tracks whose mask is already bright (flicker 0.27 -> 0.37). Known: a brief slab artifact (~3 frames) as the mouth opens.
- recordings/2026-09-22_22-56-18_raw.mp4 was zoomed in on the mouth (OBS digital crop): a face was detected in only 7/638 frames, so it can't be used. The FaceLandmarker needs eyes to chin in frame, and digital zoom adds no detail. The preview now shows a red NO FACE DETECTED label after 15 frames without a face.
- Next: Known weak spot: very wide dark regions (several adjacent missing teeth, wider than max_gap) aren't filled.
- Both opencv-python and opencv-contrib-python are in the venv; they can conflict (keep one).
