import cv2
import matplotlib.pyplot as plt
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import numpy as np
import torch
import os
from src.multilabeller.contour import Contour


class SegmentAnything:
    def __init__(self, image_input):
        #self.image = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
        self.image = image_input

        sam_checkpoint = "C:/Users/higor/Documents/projetos/multilabeller/src/multilabeller/SAM/sam_vit_b_01ec64.pth"
        model_type = "vit_b"

        device = "cuda"

        # configura o modelo do Sam
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)

        # gera a mascara
        mask_generator = SamAutomaticMaskGenerator(sam)
        self.masks = mask_generator.generate(self.image)
        self.contour_id = 0
        self.contours = []
        self.contour = None
        self.save_contours()

    def save_contours(self):
        if len(self.masks) == 0:
            return

        # Define uma imagem sobreposta vazia onde serão desenhados os contornos
        ex_mask = self.masks[0]["segmentation"]
        img_h, img_w = ex_mask.shape[0:2]
        overlay_img = np.ones((img_h, img_w, 4))
        overlay_img[:, :, 3] = 0

        # gera contornos para cada segmentação gerada
        for each_gen in self.masks:
            # Gera os contornos baseado nas mascaras em formato boolean
            boolean_mask = each_gen["segmentation"]
            uint8_mask = 255 * np.uint8(boolean_mask)
            mask_contours, _ = cv2.findContours(uint8_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            self.contour_id += 1
            self.contour = Contour(self.contour_id)

            coordinates_array = mask_contours[0]
            coordinates_list = coordinates_array.reshape(-1, 2).tolist()
            for item in coordinates_list:
                self.contour.add_contour_points(item, None)
            self.contours.append(self.contour)

    def show_outlines(self):
        if len(self.masks) == 0:
            return

        # Define uma imagem sobreposta vazia onde serão desenhados os contornos
        ex_mask = self.masks[0]["segmentation"]
        img_h, img_w = ex_mask.shape[0:2]
        overlay_img = np.ones((img_h, img_w, 4))
        overlay_img[:, :, 3] = 0

        # gera contornos para cada segmentação gerada
        for each_gen in self.masks:

            # Gera os contornos baseado nas mascaras em formato boolean
            boolean_mask = each_gen["segmentation"]

            uint8_mask = 255 * np.uint8(boolean_mask)
            mask_contours, _ = cv2.findContours(uint8_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            print(mask_contours)
            if len(mask_contours) == 0:
                continue

            # Desenha os contornos
            outline_opacity = 0.5
            outline_thickness = 2
            outline_color = np.concatenate([[0, 0, 1], [outline_opacity]])
            cv2.polylines(overlay_img, mask_contours, True, outline_color, outline_thickness, cv2.LINE_AA)

        # Desenha a imagem sobreposta
        ax = plt.gca()
        ax.set_autoscale_on(False)
        ax.imshow(overlay_img)

        return

    def show_anns(self):
        if len(self.masks) == 0:
            return
        sorted_anns = sorted(self.masks, key=(lambda x: x['area']), reverse=True)
        ax = plt.gca()
        ax.set_autoscale_on(False)

        img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
        img[:, :, 3] = 0
        for ann in sorted_anns:
            m = ann['segmentation']
            color_mask = np.concatenate([np.random.random(3), [0.35]])
            img[m] = color_mask
        ax.imshow(img)

    def plot_image(self):
        plt.figure(figsize=(20, 20))
        plt.imshow(self.image)
        # show_anns(masks)
        self.show_outlines()
        plt.axis('off')
        plt.show()
