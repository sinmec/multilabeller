import cv2
import matplotlib.pyplot as plt
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import numpy as np
import torch
import os
from src.multilabeller.contour import Contour


class SegmentAnything:
    def __init__(self, config):
        self.mask_generator = None
        self.contour_id = 0

        self.initialize(config)

    def initialize(self, config):
        model_type = config["model"]["name"]
        sam_checkpoint = config["model"]["file"]
        device = config["device"]
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        self.mask_generator = SamAutomaticMaskGenerator(sam)

    def apply(self, image_input):
        masks = self.mask_generator.generate(image_input)
        self.contours = []

        if len(masks) == 0:
            return

        ex_mask = masks[0]["segmentation"]
        img_h, img_w = ex_mask.shape[0:2]
        overlay_img = np.ones((img_h, img_w, 4))
        overlay_img[:, :, 3] = 0

        for each_gen in masks:
            boolean_mask = each_gen["segmentation"]
            uint8_mask = 255 * np.uint8(boolean_mask)
            mask_contours, _ = cv2.findContours(
                uint8_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            self.contour_id += 1
            contour = Contour(self.contour_id)

            coordinates_array = mask_contours[0]
            coordinates_list = coordinates_array.reshape(-1, 2).tolist()
            for item in coordinates_list:
                contour.add_contour_points(item, None)
            self.contours.append(contour)
