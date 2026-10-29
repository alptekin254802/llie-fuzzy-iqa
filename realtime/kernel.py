"""Measured processing path for 640x480 uint8 BGR; no camera wait or display."""
import cv2
import numpy as np
from lut_candidates import lut_candidates,make_tables
from optimized_loop import select_gamma

def process_frame(bgr,models,gammas,tables,previous):
    rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    resized=cv2.resize(rgb,(256,256),interpolation=cv2.INTER_LINEAR)
    result=lut_candidates(resized,tables,models,scorers=('fuzzy',))
    gamma=select_gamma(result,gammas,'fuzzy',previous)
    index=int(np.flatnonzero(gammas==gamma)[0])
    enhanced=cv2.LUT(bgr,tables[index])
    return enhanced,gamma
