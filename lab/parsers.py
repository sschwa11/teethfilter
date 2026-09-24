import os

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SIZE = 512
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
BISENET = {"mouth": 11, "u_lip": 12, "l_lip": 13, "skin": 1, "nose": 10, "hair": 17}
SEGFORMER = {"mouth": 10, "u_lip": 11, "l_lip": 12, "skin": 1, "nose": 2, "hair": 13}


def face_affine(pts, size=SIZE, scale=4.0):
    eye = (pts[33, :2] + pts[263, :2] + pts[133, :2] + pts[362, :2]) / 4
    mouth = (pts[78, :2] + pts[308, :2] + pts[13, :2] + pts[14, :2]) / 4
    ex = pts[263, :2] - pts[33, :2]
    ang = np.arctan2(ex[1], ex[0])
    d = np.linalg.norm(mouth - eye)
    c = 0.5 * (eye + mouth) + 0.1 * (mouth - eye)
    s = size / (scale * d)
    ca, sa = np.cos(ang) * s, np.sin(ang) * s
    m = np.array([[ca, sa, 0.0], [-sa, ca, 0.0]])
    m[:, 2] = np.array([size / 2, size / 2]) - m[:, :2] @ c
    return m


class Parser:
    def __init__(self, kind="segformer", device="cuda"):
        self.kind = kind
        self.dev = torch.device(device)
        if kind == "segformer":
            from transformers import SegformerForSemanticSegmentation
            self.net = SegformerForSemanticSegmentation.from_pretrained("jonathandinu/face-parsing").to(self.dev).eval().half()
            self.ids = SEGFORMER
        else:
            import sys
            sys.path.insert(0, os.path.join(HERE, "third_party"))
            from bisenet.model import BiSeNet
            from huggingface_hub import hf_hub_download
            self.net = BiSeNet(n_classes=19)
            self.net.load_state_dict(torch.load(hf_hub_download("vivym/face-parsing-bisenet", "79999_iter.pth"), map_location="cpu"))
            self.net = self.net.to(self.dev).eval()
            self.ids = BISENET

    @torch.no_grad()
    def probs(self, frame, pts):
        m = face_affine(pts)
        crop = cv2.warpAffine(frame, m.astype(np.float32), (SIZE, SIZE), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        x = (cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255 - MEAN) / STD
        t = torch.from_numpy(x.transpose(2, 0, 1))[None].to(self.dev)
        if self.kind == "segformer":
            logits = self.net(pixel_values=t.half()).logits.float()
            logits = torch.nn.functional.interpolate(logits, size=(SIZE, SIZE), mode="bilinear", align_corners=False)
        else:
            logits = self.net(t)[0]
        p = torch.softmax(logits, 1)[0].cpu().numpy()
        return p, m, crop

    def to_dental(self, p, m, a_dental, size=(320, 260)):
        m3 = np.vstack([m, [0, 0, 1]])
        a3 = np.vstack([a_dental, [0, 0, 1]])
        comb = (m3 @ a3)[:2].astype(np.float32)
        out = {}
        for name, idx in self.ids.items():
            out[name] = cv2.warpAffine(p[idx], comb, size, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return out
