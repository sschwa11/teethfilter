import glob
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import teeth_filter as T

CACHE = os.path.join(HERE, "cache")


def sources():
    vids = sorted(glob.glob(os.path.join(ROOT, "recordings", "*_raw.mp4")))
    vids += [os.path.join(os.path.expanduser("~"), "Videos", n) for n in (
        "2026-09-22 18-36-31.mp4", "2026-09-22 18-54-22.mp4", "2026-09-22 23-02-18.mp4", "2026-09-23 04-53-09.mp4")]
    return [v for v in vids if os.path.exists(v)]


def key_of(path):
    return os.path.splitext(os.path.basename(path))[0].replace(" ", "_")


def make_landmarker():
    import mediapipe as mp
    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision as mpv
    opts = mpv.FaceLandmarkerOptions(
        base_options=mpp.BaseOptions(model_asset_path=T.MODEL_PATH),
        running_mode=mpv.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
    )
    return mp, mpv.FaceLandmarker.create_from_options(opts)


def run(path):
    out = os.path.join(CACHE, key_of(path) + ".npz")
    if os.path.exists(out):
        return out
    mp, lm = make_landmarker()
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    pts, mats, bls, ok = [], [], [], []
    n = 0
    while True:
        good, frame = cap.read()
        if not good:
            break
        h, w = frame.shape[:2]
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        res = lm.detect_for_video(img, int(n * 1000 / fps) + 1)
        if res.face_landmarks:
            pts.append([[p.x * w, p.y * h, p.z * w] for p in res.face_landmarks[0]])
            mats.append(np.array(res.facial_transformation_matrixes[0]))
            bls.append([b.score for b in res.face_blendshapes[0]])
            ok.append(True)
        else:
            pts.append(np.zeros((478, 3)))
            mats.append(np.eye(4))
            bls.append(np.zeros(52))
            ok.append(False)
        n += 1
    cap.release()
    lm.close()
    np.savez_compressed(out, pts=np.array(pts, np.float32), mats=np.array(mats, np.float32), bls=np.array(bls, np.float32),
                        ok=np.array(ok), fps=fps, size=np.array([w, h]), path=path)
    print(f"{key_of(path)}: {n} frames, {int(np.sum(ok))} faces", flush=True)
    return out


if __name__ == "__main__":
    os.makedirs(CACHE, exist_ok=True)
    for p in sources():
        run(p)
