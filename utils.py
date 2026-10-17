import numpy as np


def normalize(row):
    """63 raw numbers -> 63 numbers that describe only the hand SHAPE."""
    pts = np.array(row, dtype=float).reshape(21, 3)
    pts = pts - pts[0]                                  # wrist becomes (0,0,0): screen position no longer matters
    scale = np.linalg.norm(pts[:, :2], axis=1).max()    # biggest wrist-to-point distance = hand size
    if scale == 0:
        scale = 1
    return (pts / scale).flatten()                      # divide by size: distance from camera no longer matters
