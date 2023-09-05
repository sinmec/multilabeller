import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow import keras
import tensorflow as tf
# tf.config.threading.set_intra_op_parallelism_threads(10)
# tf.config.threading.set_inter_op_parallelism_threads(10)

import random
import numpy as np
import cv2
import h5py
import keyboard

import sys

import config_ann as cfg_ann
from apply_UNET_mask_split import apply_UNET_mask
from divide_image import divide_image
from recreate_UNET_image import recreate_UNET_image
from find_contours import find_contours

from drawed_contour import drawed_contour
from circle_contour import circle_contour
from ellipse_contour import ellipse_contour
from drawed_contour_collection import drawed_contour_collection
from select_and_delete_contour import select_and_delete_contour
from add_small_contours import add_small_contours
from add_contours import add_contours
from add_ellipse_contours import add_ellipse_contours


# Defining the folder where the images are located and the experiment name
image_file_folder = "/home/sinmec/Documentos/imgs/"
experiment_name = os.path.basename(image_file_folder)

# Dataset prefix
dataset_prefix = '2023_08_09'

# It is possible to pre-process/pre-anottate the image with a previously
# trained UNET, thats what the variable/flag below defines
UNET_preprocessing = False
UNET_bin_trehsold = 120

# UNET model name used for pre-classifying the images
# Only important when the UNET_preprocessing is defined as True
if UNET_preprocessing:
    # model_UNET_name = "UNET_best_model_mini_04.h5"
    model_UNET_name = "../UNET_TF/UNET_best_TF.h5"
    model_UNET  = keras.models.load_model(model_UNET_name, compile=False)

# Name of the cv2 windows of the U-Net annotation app.
# The name is defined in config.py file
# Em funcao da gambi, temos um cfg_ann aqui.
window_name = cfg_ann.WINDOW_NAME

# Number of frame "jumps" in a video sequence. It's importat to have diverse samples
N_jump = 1

# Option to appply a impeller mask in the image
useMask = False

if useMask == True:
    # Loading the impeller mask
    # static_mask_img_1 = cv2.imread("/home/epic/Dropbox/impeller_mask/new_mask_from_CAD.png",0)
    static_mask_img_1 = cv2.imread("/home/epic/Dropbox/impeller_mask_white/mask_remove_background.png",0)
    static_mask_img = static_mask_img_1
else:
    static_mask_img = np.zeros_like(cfg_ann.IMG_SIZE, dtype=np.uint8)

# Just simple debug/visualisation lines
# cv2.namedWindow("test", cv2.WINDOW_NORMAL)
# cv2.imshow("test", np.hstack((static_mask_img_1, static_mask_img_2, static_mask_img_3)))
# cv2.waitKey(0)
# exit()

# U-Net related parameter. Please update in the condig.py file
stride_division = cfg_ann.STRIDE_DIVISION

# Defining the image variable and reading the files
image_files = os.listdir(image_file_folder)

# Reading the images
imgs_global = []
for img in image_files:
    if not img.endswith(".jpg"):
        continue
    imgs_global.append(img)

imgs_global.sort()
print(imgs_global)

# "Shuffling the images"
N_imgs = len(imgs_global)
random_array = np.arange(N_imgs)
# np.random.shuffle(random_array)

# Starting the image loop Here
for index_imgs in random_array:

    # if random_array[i_r] == -1:
        # print('repeating')
    # random_array[i_r] = -1

    img_file = imgs_global[index_imgs]
    load_data_from_hdf = False
    contour_collection = drawed_contour_collection()
    if index_imgs == 0:
        h5_file = h5py.File("%s_dset_%s.h5" % (dataset_prefix, experiment_name),'w')
    else:
        h5_file = h5py.File("%s_dset_%s.h5" % (dataset_prefix, experiment_name),'a')
    h5_group_name = "%s/%s" % (experiment_name, img_file)

    print(h5_group_name)


    # Creating h5 group
    h5_file.create_group(h5_group_name)

    # Creating the visualisation window
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Reading the raw image file
    img = cv2.imread(os.path.join(image_file_folder, img_file))
    img_grey = img[:,:,0]
    img_raw = img.copy()
    image_rgb = img

    # Reading the next image file (i+1)
    # img_next = cv2.imread(os.path.join(image_file_folder, imgs_global[index_imgs+1]))
    img_next = np.zeros_like(img)

    # Reading the back image file (i-1)
    img_back = cv2.imread(os.path.join(image_file_folder, imgs_global[index_imgs-1]))

    # Here I am finding the contours from the U-Net model that I specified in the first lines.
    # When we are starting a new U-Net, this must be removed
    if not load_data_from_hdf:

        if UNET_preprocessing == True:
            h_orig, w_orig, _ = image_rgb.shape


            # Applying UNET model
            subdivided_image_raw, global_coord_reference = divide_image(cfg_ann.IMG_SIZE[0], image_rgb, stride_division)
            subdivided_image_UNET = apply_UNET_mask(subdivided_image_raw, model_UNET)
            UNET_image, UNET_image_LOV = recreate_UNET_image(subdivided_image_UNET, cfg_ann.IMG_SIZE[0], image_rgb, stride_division)

            # UNET_image_LOV[UNET_image_LOV > 100] = 255
            # cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            # cv2.imshow("test", UNET_image_LOV)
            # cv2.waitKey(0)

            if useMask == True:
                UNET_image[static_mask_img>10] = 0
                UNET_image_LOV[static_mask_img>10] = 0

            UNET_image[...] = 0
            UNET_image[UNET_image_LOV > UNET_bin_trehsold] = 255
            img_contours = find_contours(UNET_image)
        else:
            img_contours = []
    else:
        # I added those lines for the case when a .h5 file is reloaded.
        # Again, it does not work 100% and I prefer to avoid this situations!
        h5_img_contours = h5_file[h5_group_name]["contours"]
        img_contours = []
        for contour in h5_img_contours:
            # _contour = h5_file[h5_group_name]["contours"][contour].value
            _contour = h5_file[h5_group_name]["contours"][contour][:]
            img_contours.append(_contour)
        del h5_file[h5_group_name]
        h5_file.create_group(h5_group_name)

    # cv2.namedWindow("test", cv2.WINDOW_NORMAL)
    # cv2.imshow("test", UNET_image_LOV)
    # cv2.waitKey(0)


    # Mapping the cv2 contours to objects ttat can be "mouse selected"
    # All of the contours starts as valid. Then those need to classified
    img_map_ID = np.zeros_like(img_grey, dtype=np.int16)
    img_map_ID[...] = -1
    for img_contour in img_contours:
        # Small contours are classified as invalid
        contour_area = cv2.contourArea(img_contour)
        if contour_area < 40:
            continue

        cv2.drawContours(img, [img_contour],-1, (0,0,255), 1)
        new_contour = drawed_contour()
        new_contour.cv2_contour = img_contour
        new_contour.valid = True
        contour_collection.add_contour(new_contour)

    # Now we are assigning an unique ID to each contour.
    # You can see the IDs in the img_map_ID variable
    for contour in contour_collection.contours:
        cv2_contour = contour.cv2_contour
        cnt_ID = contour.ID
        cv2.drawContours(img_map_ID, [cv2_contour],-1, cnt_ID, -1)
        # cv2.drawContours(img, [cv2_contour], -1, (0,255,0), 1)
    # plt.imshow(img_map_ID)
    # plt.show()

    # Storing the contour information
    contour_collection.add_map_ID(img_map_ID)

    # There are two flags/options here:
    # operationMode is the 'remove/add' contours modes:
    # operationMode = True  ---> selected contours are removed
    # operationMode = False ---> mode where you can add/annotate new contours
    operationMode = False

    # The addMode is the type of contour to be added. There are two modes
    # addMode = "contour" ---> the contour is composed of a set o mouse selected points
    # addMode = "circle"  ---> the contour is approximated by a circle, that you can control
    #                          through the mouse the size and position
    # addMode = "ellipse" ---> the contour is approximated by an ellipse, that you can control
    #                          through the mouse the size and position
    addMode = None


    # The show_mode variable controls which image to show
    # When dealing with piv images, is good to have a 'movement' idea
    # to know if we are seeing tracer particles or dispersed drops
    show_mode = 'current_image'
    gif_toggle = False
    N_toggle = 0
    while(1):


        if show_mode == 'back_image':
            img = img_next.copy()
        elif show_mode == 'current_image':
            img = img_raw.copy()
        elif show_mode =='next_image':
            img = img_back.copy()
        # elif show_mode =='halo_image':
            # img = UNET_halo_img.copy()
        elif show_mode == 'gif':
            if gif_toggle:
                img = img_raw.copy()
                cv2.waitKey(200)
            else:
                img = img_back.copy()
                cv2.waitKey(200)
            gif_toggle = ~gif_toggle
            N_toggle += 1
            if N_toggle > 10:
                N_toggle = 0
                show_mode = 'current_image'

        if operationMode:
            cv2.setMouseCallback(window_name, select_and_delete_contour, contour_collection)
            display_color = (0,0,255)
        else:
            display_color = (0,255,0)
            if addMode == "contour":
                cv2.setMouseCallback(window_name, add_contours, contour_collection)
                contour = contour_collection.contours[-1]
                pts = contour.points
                if len(pts) > 0:
                    for p in range(len(pts)):
                        x = pts[p][0]; y = pts[p][1];
                        cv2.circle(img,(x,y),3,(255,0,0),-1)
                if contour_collection.draw_contour:
                    contour = contour_collection.contours[-1]
                    print('Drawing and saving contourd %04d' % contour.ID)
                    contour.list_to_contour()
                    contour.valid = True
                    cv2.drawContours(contour_collection.map_ID, [contour.cv2_contour],-1, contour.ID, -1)
                    contour_collection.draw_contour = False
            if addMode == "circle":
                cv2.setMouseCallback(window_name, add_small_contours, [contour_collection, img])
                contour = contour_collection.contours[-1]
                # if key == 57:
                if key == 44: # , key
                    contour.change_radius_down()
                if key == 46: # . key
                    contour.change_radius_up()
                if (contour.c_x > 0) and (contour.c_y > 0):
                        cv2.circle(img,(contour.c_x,contour.c_y),contour.radius,(0,255,255),1)
                if contour_collection.draw_contour:
                    contour = contour_collection.contours[-1]
                    print('Drawing and saving contourd %04d' % contour.ID)
                    contour.list_to_contour()
                    contour.valid = True
                    cv2.drawContours(contour_collection.map_ID, [contour.cv2_contour],-1, contour.ID, -1)
                    # cv2.circle(contour_collection.map_ID, (contour.c_x,contour.c_y),contour.radius,contour.ID,-1)
                    contour_collection.last_radius = contour.radius
                    contour_collection.draw_contour = False
            if  addMode == "ellipse":
                contour = contour_collection.contours[-1]
                cv2.setMouseCallback(window_name, add_ellipse_contours, [contour_collection, img])
                for k in range(len(contour.d_1_points)):
                    _x = contour.d_1_points[k][0]
                    _y = contour.d_1_points[k][1]
                    cv2.circle(img,(_x,_y),1,(255,255,0),-1)
                if contour.commit:
                    cv2.circle(img,(contour.x_c, contour.y_c),1,(255,0,255),-1)
                    cv2.ellipse(img, (contour.x_c, contour.y_c), (contour.d_1, contour.d_2), contour.angle,
                                                                 0, 360, (255,0,0), 1)
                    if contour_collection.draw_contour:
                        print('Drawing and saving contour %04d' % contour.ID)
                        contour.list_to_contour()
                        contour.valid = True
                        cv2.drawContours(contour_collection.map_ID, [contour.cv2_contour],-1, contour.ID, -1)
                        contour_collection.last_d_2 = contour.d_2
                        contour_collection.draw_contour = False

        for contour in contour_collection.contours:
            if contour.valid:
                cv2_contour = contour.cv2_contour
                if contour.type == "draw":
                    cv2.drawContours(img, [cv2_contour], -1, display_color, 1)
                elif contour.type == "circle":
                   cv2.circle(img,(contour.c_x,contour.c_y),contour.radius,display_color,1)
                elif contour.type == "ellipse":
                   cv2.drawContours(img, [cv2_contour], -1, display_color, 1)

        cv2.imshow(window_name, img)
        key = cv2.waitKey(20)
        # print('Working with file %s!' % img_file)

        if key == 56: #key 2
            show_mode = 'back_image'
        if key == 53: #key 5
            show_mode = 'current_image'
        if key == 50: #key 2
            show_mode = 'next_image'
        if key == 52: #key 4
            # show_mode = 'halo_image'
            show_mode = 'gif'

        if key == 99 and (not operationMode): #key c
            print("creating a new contour")
            addMode = "contour"
            new_contour = drawed_contour()
            contour_collection.add_contour(new_contour)

        if key == 98 and (not operationMode): #key b
            print("creating a new small contour")
            addMode = "circle"
            new_contour = circle_contour()
            contour_collection.add_contour(new_contour)

        if key == 101 and (not operationMode): #key e
            print("creating a ellipse-type contours")
            addMode = "ellipse"
            new_contour = ellipse_contour()
            new_contour.d_2 = contour_collection.last_d_2
            contour_collection.add_contour(new_contour)

        if key == 101: # key e
            print('eee')

        if key == 32: # key Space
            operationMode = not operationMode
            print("Switching operation mode")
            if operationMode:
                print("Now in delete mode")
            else:
                print("Now in add mode")

        if key == 115: # key s
            print("Saving contours, raw_image and masks from %s to disk" % h5_group_name)
            h5_group = h5_file[h5_group_name]
            h5_group.attrs['experiment'] = experiment_name
            h5_group.attrs['image_file'] = img_file
            h5_group.create_dataset("original_img", data=img_raw[:,:,0], compression="gzip", chunks=True)
            h5_group.create_group("contours")
            mask_img = np.zeros_like(img_raw[:,:,0])
            for c, contour in enumerate(contour_collection.contours):
                if contour.valid:
                    cv2_contour = contour.cv2_contour
                    h5_cnt_group = h5_group["contours"]
                    h5_cnt_group.create_dataset("cnt_%06d" % c, data=cv2_contour, compression="gzip", chunks=True)
                    cv2.drawContours(mask_img, [cv2_contour], -1, (255), -1)
            h5_group.create_dataset("mask_img", data=mask_img, compression="gzip", chunks=True)
            h5_file.close()

        if key == 110: # key n
            print('Moving to the next image')
            break

        if key == 105: # key i
            print("IMAGE_FILE:", img_file)

        if key == 113: # key q
            exit()
