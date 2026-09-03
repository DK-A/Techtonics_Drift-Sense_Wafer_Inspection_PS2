"""Organizer baseline: brute-force ZNCC over the disclosed zoom/theta grid.

Used to (a) confirm ground truth is recoverable on present pairs and
(b) calibrate how separable the absent pairs are. This is deliberately the
naive method -- if it scores too well, the set is too easy.
"""
import sys
import numpy as np
import cv2


def warp_ref(ref, z, th):
    k = max(2, int(round(z)))
    r = cv2.blur(ref, (k, k))
    out = int(round(ref.shape[0] / z))
    M = cv2.getRotationMatrix2D(((ref.shape[1] - 1) / 2, (ref.shape[0] - 1) / 2), th, 1.0 / z)
    M[0, 2] += (out - 1) / 2 - (ref.shape[1] - 1) / 2
    M[1, 2] += (out - 1) / 2 - (ref.shape[0] - 1) / 2
    return cv2.warpAffine(r, M, (out, out), flags=cv2.INTER_LINEAR)


def search_pose(ref, srch, zooms=None, thetas=None):
    zooms = zooms if zooms is not None else np.arange(8.0, 12.01, 0.5)
    thetas = thetas if thetas is not None else np.arange(-5.0, 5.01, 1.0)
    best = (-2.0, None, None, None, None)
    for z in zooms:
        for th in thetas:
            tpl = warp_ref(ref, z, th)
            if tpl.shape[0] >= srch.shape[0]:
                continue
            res = cv2.matchTemplate(srch, tpl, cv2.TM_CCOEFF_NORMED)
            _, mx, _, loc = cv2.minMaxLoc(res)
            if mx > best[0]:
                cx = loc[0] + (tpl.shape[1] - 1) / 2.0
                cy = loc[1] + (tpl.shape[0] - 1) / 2.0
                best = (float(mx), float(cx), float(cy), float(z), float(th))
    return {"score": best[0], "x": best[1], "y": best[2], "scale": best[3], "theta": best[4]}
