# Teeth Filter for OBS

Detects your mouth live, finds the dark gaps where teeth should be, and fills only those gaps with color-matched teeth. Your real teeth are left untouched.

## Setup (Windows)

1. Install Python 3.12 from python.org (check "Add to PATH").
2. Double-click `run.bat`. First run creates a venv, installs packages, and downloads the face model.
3. In OBS, **remove or disable your webcam source** (only one app can open the camera).
4. OBS → Sources → **+ → Browser** → URL `http://127.0.0.1:8765/`, width 1280, height 720.
   - Right-click the source → Properties → uncheck "Shutdown source when not visible".
   - Fallback: **Window Capture** of the "Teeth Filter" window.
5. OBS → **Start Virtual Camera**, then pick "OBS Virtual Camera" in Zoom/Teams/Meet.

## Choosing a webcam

- First launch with more than one camera shows a numbered list; pick one and it's remembered next time.
- `run.bat --pick` shows the list again.
- `run.bat --list-cameras` just prints the list.
- `run.bat --camera-name "brio"` selects by (partial) name.
- Press `c` in the preview window to switch to the next camera live.
- OBS Virtual Camera is listed too (marked "OBS output"). If you feed the filter from it, point OBS's virtual camera at a scene/source that doesn't include the Teeth Filter browser source.

## Using your own teeth (best realism)

Find an older photo of yourself smiling with full teeth, facing the camera, and run:

```
run.bat --ref "C:\path\to\old_smile.jpg"
```

Press `p` to switch the fill source to `reference`.

## How it works

The filter tracks your head and knows where your missing teeth sit relative to your skull, so it never has to hunt for dark gaps. Each frame it locks onto the edge of your real front tooth, mirrors your own teeth from the other side of your mouth into the missing positions (upper and lower), and keeps your lips in front using a small mouth-segmentation model trained on your recordings. Everything comes from the live frame, so lighting and focus always match.

It needs `teeth_model.npz` and `teeth_seg.onnx` (both included). The older gap-detecting methods are still available with `--method edge` or `--method legacy`.

See **HANDOFF.md** for current quality numbers, known problems, how to retrain after recording new clips, and how to keep working on it.

## AI fill (optional, recommended)

A small neural network trained on your own recordings repaints the missing teeth, instead of mirror-copying a neighbor tooth.

1. Record 3-5 clips with `r` (talk, smile, laugh; whole face in frame, good light). More variety = better.
2. Double-click `train_ai.bat`. The first run installs PyTorch with CUDA (~3 GB), then:
   - builds a dataset from every `recordings\*_raw.mp4` (a few minutes),
   - trains on your RTX GPU (~15-30 min),
   - saves preview images to `ai_preview\` (left: original, middle: hidden area, right: AI fill; the lower rows are your real gaps),
   - writes `teeth_ai.onnx`.
3. Restart `run.bat`. The filter uses the AI fill automatically (runs on the GPU via DirectML). Press `m` to switch between AI and mirror fill.

After recording new clips, retrain with `train_ai.bat --rebuild`. Longer training: `train_ai.bat --iters 16000`.

## Auto-calibration

Press `a` in the preview window (or start with `run.bat --auto`). Face the camera and talk and smile normally for 20-30 seconds. The filter learns where your missing teeth are and how far your teeth reach toward the mouth corners. After about 150 good frames it starts ignoring dark spots in places where it has never seen a missing tooth. Press `a` again to stop; everything is saved to `settings.json` and loaded next time. `x` clears the learned gap positions.

## Recording test clips

Press `r` to start/stop recording. Two files are saved in the `recordings` folder: `*_raw.mp4` (camera, unfiltered) and `*_filtered.mp4`. Send the raw one when reporting problems: it can be re-run through the filter offline with `run.bat --video recordings\<name>_raw.mp4`.

The preview window opens at 852x480 (`--window-width` / `--window-height` to change). Capture stays at 1280x720 for tracking quality.

## Tuning (click the preview window first)

| Key | Action |
|---|---|
| `m` | AI fill / mirror fill |
| `a` | Auto-calibration on/off |
| `r` | Record raw + filtered clips |
| `x` | Clear learned gap positions |
| `c` | Next webcam |
| `h` | Show/hide settings overlay (the OBS feed never shows it) |
| `d` | Debug: red = pixels being replaced |
| `f` | Filter on/off (before/after check) |
| `[` `]` | Widest gap that can be filled |
| `-` `=` | Narrowest gap that counts as missing (raise if normal spaces between teeth get filled) |
| `,` `.` | How far toward the mouth corners to look |
| `u` `l` | Upper / lower teeth on/off |
| `p` | Fill source: self (clone your teeth) / reference / procedural |
| `q` | Quit |

Settings are saved to `settings.json` on quit. Command-line flags (`--max-gap`, `--min-gap`, `--extent`, `--min-open`) override saved values.

## Tips

- Even, frontal lighting (a lamp behind the monitor) makes detection much more stable.
- Test it: record yourself in OBS talking and smiling for a minute with `f` toggled, and check playback before an interview.
- Test a single photo: `run.bat --image photo.jpg --out result.png`
