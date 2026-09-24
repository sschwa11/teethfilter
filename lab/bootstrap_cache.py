import os

import numpy as np

import common as C

if __name__ == "__main__":
    os.makedirs(C.CACHE, exist_ok=True)
    z = np.load(os.path.join(os.path.dirname(C.HERE), "teeth_model.npz"))
    np.save(os.path.join(C.CACHE, "ref_shape.npy"), z["ref_shape"])
    np.save(os.path.join(C.CACHE, "rigid_idx.npy"), z["rigid_idx"])
    print("wrote cache/ref_shape.npy and cache/rigid_idx.npy from teeth_model.npz")
