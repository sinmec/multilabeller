import cv2
import numpy as np
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

from src.multilabeller.SAM_contour import SAM_Contour


class SegmentAnything:
    def __init__(self, config):
        self.mask_generator = None
        self.image_manipulator = None
        self.initialize(config)
        self.contours = []

    def initialize(self, config):
        model_type = config["model"]["name"]
        sam_checkpoint = config["model"]["file"]
        device = config["device"]
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        self.mask_generator = SamAutomaticMaskGenerator(sam)

    def apply(self, image_input):
        masks = self.mask_generator.generate(image_input)

        mask_contours = []

        if len(masks) == 0:
            return

        ex_mask = masks[0]["segmentation"]
        img_h, img_w = ex_mask.shape[0:2]
        overlay_img = np.ones((img_h, img_w, 4))
        overlay_img[:, :, 3] = 0

        for each_gen in masks:
            boolean_mask = each_gen["segmentation"]
            uint8_mask = 255 * np.uint8(boolean_mask)
            mask_contour, _ = cv2.findContours(
                uint8_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )

            x_coordinates = mask_contour[0][:, 0, 0]
            y_coordinates = mask_contour[0][:, 0, 1]

            if np.nanmin(x_coordinates) == 0:
                continue
            if np.nanmin(y_coordinates) == 0:
                continue
            if np.nanmax(x_coordinates) > img_w - 2:
                continue
            if np.nanmax(y_coordinates) > img_h - 2:
                continue

            mask_contours.append(mask_contour)

        for mask_contour in mask_contours:
            SAM_contour = SAM_Contour()

            coordinates_array = mask_contour[0]
            coordinates_list = coordinates_array.reshape(-1, 2).tolist()
            for point in coordinates_list:
                SAM_contour.add_points(point)
            SAM_contour.translate_from_annotation_to_navigation_windows(
                self.image_manipulator
            )
            SAM_contour.to_cv2_contour()
            SAM_contour.in_progress = False
            SAM_contour.finished = True

            self.contours.append(SAM_contour)
