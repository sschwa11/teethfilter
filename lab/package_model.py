import os
import shutil
import sys

import numpy as np

import common as C

if __name__ == "__main__":
    seg = sys.argv[1] if len(sys.argv) > 1 else "seg_w16"
    root = os.path.dirname(C.HERE)
    z = np.load(os.path.join(C.CACHE, "model_v10c.npz"))
    np.savez_compressed(os.path.join(root, "teeth_model.npz"), ref_shape=np.load(os.path.join(C.CACHE, "ref_shape.npy")),
                        rigid_idx=np.load(os.path.join(C.CACHE, "rigid_idx.npy")), x_mid=z["x_mid"], edge_offset=z["edge_offset"],
                        bottom=z["bottom"], slot=z["slot"].astype(np.float16), donor_mask=z["donor_mask"].astype(np.uint8))
    shutil.copy(os.path.join(C.CACHE, seg + ".onnx"), os.path.join(root, "teeth_seg.onnx"))
    print("wrote teeth_model.npz and teeth_seg.onnx")
