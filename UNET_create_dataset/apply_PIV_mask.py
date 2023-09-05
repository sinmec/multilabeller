import cv2
import numpy as np

def apply_PIV_mask(img_folder, img_name, c_x, c_y, d_theta):
    mask_volute = cv2.imread('/home/epic/Data/EPIC_PIV/PIV_masks_new/output/300_rpm_760_kgh_1bar/enc_00/masks/mask_volute.png',0)
    mask_blades = cv2.imread('/home/epic/Data/EPIC_PIV/PIV_masks_new/output/300_rpm_760_kgh_1bar/enc_00/masks/mask_impeller.png',0)
    mask_blades_base = cv2.imread('/home/epic/Data/EPIC_PIV/PIV_masks_new/output/300_rpm_760_kgh_1bar/enc_00/masks/mask_impeller_base.png',0)
    img_piv = cv2.imread('/home/epic/EXPERIMENTS//PIV_oil_in_water/%s/%s' % (img_folder, img_name),0)

    mask_blades_base = cv2.bitwise_not(mask_blades_base)

    h, w = img_piv.shape[:2]

    # mask_blades = cv2.bitwise_not(mask_blades)
    keep_running = True
    M = cv2.getRotationMatrix2D((c_x, c_y), d_theta,1)
    mask_blade_rot = cv2.warpAffine(mask_blades.copy(),M, (w, h), 0)

    # mask = mask_blade_rot + mask_volute
    mask = mask_blades_base + mask_volute
    mask = cv2.bitwise_not(mask)

    return mask

