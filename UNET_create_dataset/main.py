import os

import cv2
import h5py
import numpy as np

from add_contours import add_contours
from add_ellipse_contours import add_ellipse_contours
from add_small_contours import add_small_contours
from circle_contour import circle_contour
from drawed_contour import drawed_contour
from drawed_contour_collection import drawed_contour_collection
from ellipse_contour import ellipse_contour
from select_and_delete_contour import select_and_delete_contour

# Defining the folder where the images are located and the experiment name
image_file_folder = r"C:\Users\rafaelfc\Data\multilabeller\imgs"
experiment_name = os.path.basename(image_file_folder)

# Dataset prefix
dataset_prefix = "2023_08_09"

# Name of the cv2 windows of the U-Net annotation app.
window_name = "TEST"

# Number of frame "jumps" in a video sequence. It's importat to have diverse samples
N_jump = 1

# Defining the image_manipulator variable and reading the files
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

# Starting the image_manipulator loop Here
for index_imgs in random_array:
    img_file = imgs_global[index_imgs]
    load_data_from_hdf = False
    contour_collection = drawed_contour_collection()
    if index_imgs == 0:
        h5_file = h5py.File("%s_dset_%s.h5" % (dataset_prefix, experiment_name), "w")
    else:
        h5_file = h5py.File("%s_dset_%s.h5" % (dataset_prefix, experiment_name), "a")
    h5_group_name = "%s/%s" % (experiment_name, img_file)

    print(h5_group_name)

    # Creating h5 group
    h5_file.create_group(h5_group_name)

    # Creating the visualisation window
    cv2.namedWindow(window_name, cv2.WINDOW_GUI_NORMAL)

    # Reading the raw image_manipulator file
    img = cv2.imread(os.path.join(image_file_folder, img_file))
    img_grey = img[:, :, 0]
    img_raw = img.copy()
    image_rgb = img

    # Reading the next image_manipulator file (i+1)
    # img_next = cv2.imread(os.path.join(image_file_folder, imgs_global[index_imgs+1]))
    img = cv2.imread(os.path.join(image_file_folder, img_file))

    img_contours = []

    # Mapping the cv2 contours to objects ttat can be "mouse selected"
    # All the contours starts as valid. Then those need to classified
    img_map_ID = np.zeros_like(img_grey, dtype=np.int16)
    img_map_ID[...] = -1
    for img_contour in img_contours:
        # Small contours are classified as invalid
        contour_area = cv2.contourArea(img_contour)
        if contour_area < 40:
            continue

        cv2.drawContours(img, [img_contour], -1, (0, 0, 255), 1)
        new_contour = drawed_contour()
        new_contour.cv2_contour = img_contour
        new_contour.valid = True
        contour_collection.add_contour(new_contour)

    # Now we are assigning an unique ID to each contour.
    # You can see the IDs in the img_map_ID variable
    for contour in contour_collection.contours:
        cv2_contour = contour.cv2_contour
        cnt_ID = contour.ID
        cv2.drawContours(img_map_ID, [cv2_contour], -1, cnt_ID, -1)
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

    # The show_mode variable controls which image_manipulator to show
    # When dealing with piv images, is good to have a 'movement' idea
    # to know if we are seeing tracer particles or dispersed drops
    show_mode = "current_image"
    gif_toggle = False
    N_toggle = 0
    while 1:
        img = img_raw.copy()

        if operationMode:
            cv2.setMouseCallback(
                window_name, select_and_delete_contour, contour_collection
            )
            display_color = (0, 0, 255)
        else:
            display_color = (0, 255, 0)
            if addMode == "contour":
                cv2.setMouseCallback(window_name, add_contours, contour_collection)
                contour = contour_collection.contours[-1]
                pts = contour.points
                if len(pts) > 0:
                    for p in range(len(pts)):
                        x = pts[p][0]
                        y = pts[p][1]
                        cv2.circle(img, (x, y), 3, (255, 0, 0), -1)
                if contour_collection.draw_contour:
                    contour = contour_collection.contours[-1]
                    print("Drawing and saving contourd %04d" % contour.ID)
                    contour.list_to_contour()
                    contour.valid = True
                    cv2.drawContours(
                        contour_collection.map_ID,
                        [contour.cv2_contour],
                        -1,
                        contour.ID,
                        -1,
                    )
                    contour_collection.draw_contour = False
            if addMode == "circle":
                cv2.setMouseCallback(
                    window_name, add_small_contours, [contour_collection, img]
                )
                contour = contour_collection.contours[-1]
                if key == 44:  # , key
                    contour.change_radius_down()
                if key == 46:  # . key
                    contour.change_radius_up()
                if (contour.c_x > 0) and (contour.c_y > 0):
                    cv2.circle(
                        img,
                        (contour.c_x, contour.c_y),
                        contour.radius,
                        (0, 255, 255),
                        1,
                    )
                if contour_collection.draw_contour:
                    contour = contour_collection.contours[-1]
                    print("Drawing and saving contourd %04d" % contour.ID)
                    contour.list_to_contour()
                    contour.valid = True
                    cv2.drawContours(
                        contour_collection.map_ID,
                        [contour.cv2_contour],
                        -1,
                        contour.ID,
                        -1,
                    )
                    # cv2.circle(contour_collection.map_ID, (contour.c_x,contour.c_y),contour.radius,contour.ID,-1)
                    contour_collection.last_radius = contour.radius
                    contour_collection.draw_contour = False
            if addMode == "ellipse":
                contour = contour_collection.contours[-1]
                cv2.setMouseCallback(
                    window_name, add_ellipse_contours, [contour_collection, img]
                )
                for k in range(len(contour.d_1_points)):
                    _x = contour.d_1_points[k][0]
                    _y = contour.d_1_points[k][1]
                    cv2.circle(img, (_x, _y), 1, (255, 255, 0), -1)
                if contour.commit:
                    cv2.circle(img, (contour.x_c, contour.y_c), 1, (255, 0, 255), -1)
                    cv2.ellipse(
                        img,
                        (contour.x_c, contour.y_c),
                        (contour.d_1, contour.d_2),
                        contour.angle,
                        0,
                        360,
                        (255, 0, 0),
                        1,
                    )
                    if contour_collection.draw_contour:
                        print("Drawing and saving contour %04d" % contour.ID)
                        contour.list_to_contour()
                        contour.valid = True
                        cv2.drawContours(
                            contour_collection.map_ID,
                            [contour.cv2_contour],
                            -1,
                            contour.ID,
                            -1,
                        )
                        contour_collection.last_d_2 = contour.d_2
                        contour_collection.draw_contour = False

        for contour in contour_collection.contours:
            if contour.valid:
                cv2_contour = contour.cv2_contour
                if contour.type == "draw":
                    cv2.drawContours(img, [cv2_contour], -1, display_color, 1)
                elif contour.type == "circle":
                    cv2.circle(
                        img,
                        (contour.c_x, contour.c_y),
                        contour.radius,
                        display_color,
                        1,
                    )
                elif contour.type == "ellipse":
                    cv2.drawContours(img, [cv2_contour], -1, display_color, 1)

        cv2.imshow(window_name, img)
        key = cv2.waitKey(20)

        if key == 56:  # key 2
            show_mode = "back_image"
        if key == 53:  # key 5
            show_mode = "current_image"
        if key == 50:  # key 2
            show_mode = "next_image"
        if key == 52:  # key 4
            # show_mode = 'halo_image'
            show_mode = "gif"

        if key == 99 and (not operationMode):  # key c
            print("creating a new contour")
            addMode = "contour"
            new_contour = drawed_contour()
            contour_collection.add_contour(new_contour)

        if key == 98 and (not operationMode):  # key b
            print("creating a new small contour")
            addMode = "circle"
            new_contour = circle_contour()
            contour_collection.add_contour(new_contour)

        if key == 101 and (not operationMode):  # key e
            print("creating a ellipse-type contours")
            addMode = "ellipse"
            new_contour = ellipse_contour()
            new_contour.d_2 = contour_collection.last_d_2
            contour_collection.add_contour(new_contour)

        if key == 32:  # key Space
            operationMode = not operationMode
            print("Switching operation mode")
            if operationMode:
                print("Now in delete mode")
            else:
                print("Now in add mode")

        if key == 115:  # key s
            print(
                "Saving contours, raw_image and masks from %s to disk" % h5_group_name
            )
            h5_group = h5_file[h5_group_name]
            h5_group.attrs["experiment"] = experiment_name
            h5_group.attrs["image_file"] = img_file
            h5_group.create_dataset(
                "original_img", data=img_raw[:, :, 0], compression="gzip", chunks=True
            )
            h5_group.create_group("contours")
            mask_img = np.zeros_like(img_raw[:, :, 0])
            for c, contour in enumerate(contour_collection.contours):
                if contour.valid:
                    cv2_contour = contour.cv2_contour
                    h5_cnt_group = h5_group["contours"]
                    h5_cnt_group.create_dataset(
                        "cnt_%06d" % c,
                        data=cv2_contour,
                        compression="gzip",
                        chunks=True,
                    )
                    cv2.drawContours(mask_img, [cv2_contour], -1, (255), -1)
            h5_group.create_dataset(
                "mask_img", data=mask_img, compression="gzip", chunks=True
            )
            h5_file.close()

        if key == 110:  # key n
            print("Moving to the next image_manipulator")
            break

        if key == 105:  # key i
            print("IMAGE_FILE:", img_file)

        if key == 113:  # key q
            exit()
