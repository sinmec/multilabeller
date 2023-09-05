import numpy as np
import cv2
import matplotlib.pyplot as plt
import os


# Folder where to read the images
# image_files_folder = "/home/epic/EXPERIMENTS/exps_oil_in_water_white_20cp/exp_N300rpm_760kgh_25gs"
# image_files_folder = "/home/epic/EXPERIMENTS/exps_oil_in_water_white_20cp/exp_N300rpm_1120kgh_25gs"
image_files_folder = "/home/epic/EXPERIMENTS/exps_oil_in_water_white_20cp/exp_N300rpm_150kgh_5gs"
# image_files_folder = "/home/epic/Data/EPIC_PIV/find_angular_position_HSC/exp_N300rpm_760kgh_25gs_fixed"
# image_files_folder = "/home/epic/EXPERIMENTS/exps_water_in_oil/exp_N600rpm_BEP0.8_D1000rpm"
image_files = os.listdir(image_files_folder)

# Sorting the images...
image_files.sort()

# 1000 images are enough to extract the background
image_files = image_files[:2000]
# image_files = image_files[:200]

# Returning the total number of full images
N_full_images = len(image_files)

# Creating the matrices that store the images
# image_all = np.zeros((N_full_images, 1536, 1536), dtype=float)
image_tmp = np.zeros((1536, 1536), dtype=float)
for k in range(N_full_images):
    if k % 100 == 0:
        print(k, N_full_images)

    # Reading the image and resizing to a fized 1536 x 1536 image size
    # This is done to follown the original function implementation
    image = cv2.imread(os.path.join(image_files_folder, image_files[k]), 0)
    h_orig, w_orig = image.shape
    image = cv2.resize(image, (1536 ,1536))

    # Turning the image to a float, to allow better average calculation
    image = image.astype(float)

    # Inserting the image to a vector
    # PS: You may compute the average without storing the values on an
    #     intermediate matrix. Here I am using to approach since it allows
    #     a better flexibility
    # image_all[k,:,:] = image
    # p_expoent = -100.0
    p_expoent = 1.0
    image_tmp += np.power(image, p_expoent)

# Computing the mean (or other desired field) from the stored images
# image_all_mean = np.median(image_all, axis=0)
# image_all_mean = np.mean(image_all, axis=0)

# Min/Mean/Max approach from DantecStudio software
# p_expoent = -100.0
# image_all_power  = np.power(image_all, p_expoent)
# image_all_power  = np.sum(image_all_power, axis=0)
# image_all_power /= float(N_full_images)
# image_all_power  = np.power(image_all_power, (1.0 / p_expoent))
# image_all_mean = image_all_power
# print(image_all_power.shape)
# exit()

image_all_power = image_tmp / float(N_full_images)
image_all_power = np.power(image_all_power, (1.0 / p_expoent))
image_all_mean = image_all_power

# Transforming it back to an unsigned 8-bit integer type
image_background = image_all_mean.astype(np.uint8)

# Resizing to the original shape
image_background = cv2.resize(image_background, (w_orig, h_orig))

## Displaying the background image
#cv2.namedWindow("test", cv2.WINDOW_NORMAL)
#cv2.imshow("test", image_background)
#cv2.waitKey(0)

cv2.imwrite('new_background_%s.jpg' % image_files_folder.split('/')[-1], image_background)

#### Repeating the loop to debug/visualize the new images
##for k in range(N_full_images):
##   image = cv2.imread(os.path.join(image_files_folder, image_files[k]), 0)
##
##   image            = image.astype(float)
##   image_background = image_background.astype(float)
##
##   img_final = np.abs(image - image_background)
##   img_final = img_final.astype(np.uint8)
##
##   cv2.namedWindow("test", cv2.WINDOW_NORMAL)
##   cv2.imshow("test", img_final)
##   cv2.waitKey(0)

