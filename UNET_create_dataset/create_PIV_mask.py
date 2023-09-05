import cv2
import numpy as np

c_x= 1209; c_y=701

imgs_info = np.genfromtxt('img_list.txt', delimiter=',', dtype='str')

for img_info in imgs_info:
    img_folder = img_info[0]
    img_name   = img_info[1]
    print(img_folder)
    print(img_name  )
    mask_volute = cv2.imread('/home/epic/Data/EPIC_PIV/PIV_masks_new/output/300_rpm_760_kgh_1bar/enc_00/masks/mask_volute.png',0)
    mask_blades = cv2.imread('/home/epic/Data/EPIC_PIV/PIV_masks_new/output/300_rpm_760_kgh_1bar/enc_00/masks/mask_impeller.png',0)
    img_piv = cv2.imread('/home/epic/EXPERIMENTS//PIV_oil_in_water/%s/%s' % (img_folder, img_name),0)

    h, w = img_piv.shape[:2]

    # mask_blades = cv2.bitwise_not(mask_blades)
    d_theta = 0
    keep_running = True
    while(keep_running):
        M = cv2.getRotationMatrix2D((c_x, c_y), d_theta,1)
        mask_blade_rot = cv2.warpAffine(mask_blades.copy(),M, (w, h), 0)

        mask = mask_blade_rot + mask_volute
        mask = cv2.bitwise_not(mask)

        img_piv_masked = cv2.bitwise_and(mask, img_piv.copy())

        # cv2.imshow('angle_test', np.hstack((mask_volute, mask_blade_rot, mask)))

        cv2.namedWindow('angle_test', cv2.WINDOW_NORMAL)
        cv2.imshow('angle_test', img_piv_masked)
        key = cv2.waitKey(0)
        if key == 43: # key +
            d_theta += 0.5
        elif key == 43: # key -
            d_theta -= 0.5
        elif key == 115: # key s
            theta = d_theta
            np.savetxt('%s.txt' % img_name, np.c_[c_x, c_y, theta], delimiter=',', comments='')
            keep_running = False

