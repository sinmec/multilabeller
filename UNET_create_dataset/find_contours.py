import os
import cv2
import numpy as np
import tensorflow as tf
import time


def find_contours(img_UNET):

    # Crearing the contour list
    img_contours = []

    # Finding the contours
    contours, hierarchy = cv2.findContours(img_UNET,
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)
    # Total number of contours
    N_contours = len(contours)

    # If no contours, return an empty list
    if N_contours < 1:
        return []

    # Looping around the contours
    for cnt in contours:
        X = cnt[:,0,0]
        Y = cnt[:,0,1]

        # If the contour touch the boundaries, it is not valid
        l_flag = np.min(X) == 0
        r_flag = np.max(X) == (img_UNET.shape[0] - 1)
        t_flag = np.min(Y) == 0
        b_flag = np.max(Y) == (img_UNET.shape[1] - 1)

        if not (l_flag + r_flag + t_flag + b_flag):
            img_contours.append(cnt)

    # Returning the contours
    return img_contours
