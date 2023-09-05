import numpy as np
import pandas as pd
import os
import cv2


"""dataset_UNET_2_P12
   This script generates the "real" UNET dataset folder, which is composed
   of smaller 'sub-images'. They can be generated systematically or in a
   random fashion
"""

# Dataset folder
dataset_folder = "dataset_TNT_flow"

# Size of the sub-images
window_size = 100

# Random mode option
random_mode = True
# random_samples = int(4.0 * (1600 // window_size))
# random_samples *= random_samples
random_samples = 200

N_VALIDATION = 2

# We have the Validation and the Training folders
cwd = os.path.dirname(os.path.abspath(__file__))
for _folder in ["Validation", "Training"]:

    # Creating the output folders
    OUTPUT_FOLDER = os.path.join(cwd,dataset_folder, _folder)
    IMAGES_FOLDER = os.path.join(cwd, dataset_folder, "images_full")
    MASKS_FOLDER  = os.path.join(cwd, dataset_folder, "masks_full")
    IMAGES_SPLIT_FOLDER = os.path.join(OUTPUT_FOLDER,"images")
    MASKS_SPLIT_FOLDER  = os.path.join(OUTPUT_FOLDER,"masks")

    try:
        os.makedirs(OUTPUT_FOLDER)
    except:
        print('images folder already exist')

    try:
        os.makedirs(IMAGES_SPLIT_FOLDER)
    except:
        print('images folder already exist')

    try:
        os.makedirs(MASKS_SPLIT_FOLDER)
    except:
        print('masks folder already exist')


    # Listing the images
    imgs = os.listdir(IMAGES_FOLDER)
    imgs.sort()

    for img_file in imgs:

        img = cv2.imread(os.path.join(IMAGES_FOLDER, img_file),0)
        img_mask = cv2.imread(os.path.join(MASKS_FOLDER, img_file),0)

        N_I = img.shape[0] // window_size
        N_J = img.shape[1] // window_size
        random_cnt = 0
        if random_mode:
            for n in range(random_samples):

                rand_i = np.random.randint(low=0, high= img.shape[0] - window_size)
                rand_j = np.random.randint(low=0, high= img.shape[1] - window_size)

                img_split  =      img[rand_i:(rand_i+window_size), rand_j:(rand_j+window_size)]
                mask_split = img_mask[rand_i:(rand_i+window_size), rand_j:(rand_j+window_size)]

                # print('Bagunca no Validationooo')
                if "Validation" in IMAGES_SPLIT_FOLDER:
                    if random_cnt < N_VALIDATION:
                        cv2.imwrite(os.path.join(IMAGES_SPLIT_FOLDER,"%s_%03d.jpg" % (img_file, random_cnt)), img_split)
                        cv2.imwrite(os.path.join(MASKS_SPLIT_FOLDER ,"%s_%03d.png" % (img_file, random_cnt)), mask_split)
                if "Training" in IMAGES_SPLIT_FOLDER:
                    if random_cnt > N_VALIDATION:
                        cv2.imwrite(os.path.join(IMAGES_SPLIT_FOLDER,"%s_%03d.jpg" % (img_file, random_cnt)), img_split)
                        cv2.imwrite(os.path.join(MASKS_SPLIT_FOLDER ,"%s_%03d.png" % (img_file, random_cnt)), mask_split)
                random_cnt += 1

        else:
            for i in range(N_I):
                for j in range(N_J):
                    img_split  =  img[i*window_size:(i+1)*window_size, j*window_size:(j+1)*window_size]
                    mask_split = img_mask[i*window_size:(i+1)*window_size, j*window_size:(j+1)*window_size]

                    cv2.imwrite(os.path.join(IMAGES_SPLIT_FOLDER,"%s_%03d_%03d.jpg" % (img_file, i, j)), img_split)
                    cv2.imwrite(os.path.join(MASKS_SPLIT_FOLDER ,"%s_%03d_%03d.png" % (img_file, i, j)), mask_split)
