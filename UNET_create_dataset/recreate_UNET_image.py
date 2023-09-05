import os
import cv2
import numpy as np
import tensorflow as tf
import time

# import config as cfg

def recreate_UNET_image(subdivided_image_UNET, window_size, image_rgb, stride_division=2):


    # Only the first image channel
    raw_img = image_rgb[:, :, 0]

    # Initializing the U-Net image numerator and denominator
    # This will become clear in the next lines
    UNET_image = np.zeros_like(raw_img, dtype=float)
    UNET_image_nmax = np.zeros_like(raw_img, dtype=float)

    # Number of sliding window passses
    N_I = raw_img.shape[0] // window_size + 1
    N_J = raw_img.shape[1] // window_size + 1

    # This check the sliding consistency
    diff_I = - raw_img.shape[0] + N_I * window_size
    diff_J = - raw_img.shape[1] + N_J * window_size
    assert diff_I % 2 == 0, "diff_I is not even!"
    assert diff_J % 2 == 0, "diff_J is not even!"

    # Number os scan passes in the I and J direction
    scan_I = np.arange(0, raw_img.shape[0] - window_size + 1, window_size // stride_division)
    scan_J = np.arange(0, raw_img.shape[1] - window_size + 1, window_size // stride_division)

    # print(scan_I)
    # print(scan_J)
    # exit()

    # Generating the U-Net image 
    index = 0
    for i in scan_I:
        for j in scan_J:
            i_start = i
            i_end   = i + window_size
            j_start = j
            j_end   = j + window_size

            # Returning the 'intermediary' U-Net from the sliding window full matrix
            # See that I am summing this 'sub-image' into the original/desired U-Net image
            # That is the 'U-Net numerator'
            UNET_image[i_start:i_end, j_start:j_end] += subdivided_image_UNET[index, :, :, 0]
            # UNET_image[i_start:i_end, j_start:j_end] = subdivided_image_UNET[index, :, :, 0]


            # Since we are summing in the numerator, and the values are bounded to 0 and 255
            # I have a second matrix, the UNET_image_nmax, which is a denominator
            # Therefore, I am updating the denominator here
            UNET_image_nmax[i_start:i_end, j_start:j_end] += 255.0
            index += 1

    # print("PROBLEM WHILE GENERATING U-NET IMAGES. FIX IT!!! USE 1 STRIDE_DIV")
#    # Now I divide the image
    UNET_image /= UNET_image_nmax
    UNET_image /= np.nanmax(UNET_image)
    UNET_image *= 255.0
    UNET_image = UNET_image.astype(np.uint8)

    # Generating a binary image. Here I am defining a value,
    # but you can easily use an otsu or something a bit more automatic
    UNET_image_bin = np.zeros_like(UNET_image)
    # UNET_image_bin[UNET_image >  20] = 255
    # UNET_image_bin[UNET_image <= 20] = 0

    # print('############## APPLYING UNET EROSION  ##################')
    # kernel = np.ones((3, 3), np.uint8)
    # UNET_image_bin = cv2.erode(UNET_image_bin, kernel)

    print('############## NOT APPLYING MASK ##################')
    # Applying the impeller blade mask
    # mask_img = cv2.imread(cfg.MASK_FILE,0)
    # mask_img = cv2.resize(mask_img, (1536, 1536))
    # UNET_image_bin[mask_img > 10] = 0
    # UNET_image[mask_img > 10] = 0

    # cv2.namedWindow("teste", cv2.WINDOW_NORMAL)
    # cv2.imshow("teste", UNET_image)
    # cv2.waitKey(0)

    # Returning the binary and the U-Net original image
    return UNET_image_bin, UNET_image
