import os
import numpy as np
import cv2
import h5py
from pathlib import Path
import matplotlib.pyplot as plt


"""
    This script read the UNET .h5 dataset files from a folder and
    outputs a new folder with the UNET dataset
"""

# Defining the folder where the .h5 files are located
# dset_unet_folder = os.getcwd()
dset_unet_folder = "."
h5_files = os.listdir(dset_unet_folder)

# Define mask mode [full or halo]
mask_mode = "full"
halo_width = 2

# Name of the generated dataset folder
target_folder = "dataset_TNT_flow"

# If there is the need to remove the background img
background_subtraction = False

# Creating the new folder
Path(os.path.join(target_folder, 'masks_full')).mkdir(parents=True, exist_ok=True)
Path(os.path.join(target_folder, 'images_full')).mkdir(parents=True, exist_ok=True)

for _h5_file in h5_files:

    if not ".h5" in _h5_file:
        continue

    # Reading the .h5 file
    h5_file = os.path.join(dset_unet_folder, _h5_file)
    h5_file = h5py.File(h5_file,'r')

    # Looping over the .h5 file structure
    experiments = h5_file.keys()
    experiments = ['A'] # dummy
    for experiment in experiments:
        images = h5_file.keys()

        # Looping over the images
        for image in images:
            img_out_file = image
            if not "contours" in h5_file[image].keys():
                continue

            original_image = h5_file[image]['original_img']
            raw_img = original_image[:]
            # print(img_out_file)
            # print(raw_img.shape)
            # cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            # cv2.imshow("test", raw_img)
            # cv2.waitKey(0)
            mask_img = np.zeros_like(raw_img)

            if background_subtraction:
                image_background = cv2.imread('new_background_%s.jpg' % experiment, 0)
                raw_img          = raw_img.astype(float)
                image_background = image_background.astype(float)
                img_final = np.abs(raw_img - image_background)
                img_final = img_final.astype(np.uint8)
                raw_img = img_final

            # Looping over the contours
            contours = h5_file[image]["contours"]
            for contour in contours:
                cnt = contours[contour][:]

                if mask_mode == "full":
                    cv2.drawContours(mask_img, [cnt], -1, 255, -1)
                    cv2.drawContours(mask_img, [cnt], -1, 0, 1)
                else:
                    cv2.drawContours(mask_img, [cnt], -1, 255, halo_width)

            # print("!!!!!!CROPPING IMAGES!!!!!!!!")
            # raw_img  = raw_img[185:1214, 643:1733]
            # mask_img = mask_img[185:1214, 643:1733]

            # Writing the images
            cv2.imwrite("./%s/images_full/%s" % (target_folder, img_out_file), raw_img)
            cv2.imwrite("./%s/masks_full/%s"  % (target_folder, img_out_file), mask_img)
