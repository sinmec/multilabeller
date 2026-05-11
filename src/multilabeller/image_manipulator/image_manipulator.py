import cv2
import numpy as np


class ImageManipulator:
    def __init__(self, image, config):
        self.config = config
        self.navigation_image_original = image.copy()
        self.annotation_image_original = image.copy()
        self.i = 0
        self.navigation_image = image.copy()
        self.navigation_image_buffer = image.copy()
        self.annotation_image = image.copy()
        self.annotation_image_buffer = image.copy()

        self.navigation_image_width = 0
        self.navigation_image_height = 0
        self.navigation_image_aspect_ratio = 0

        self.annotation_image_coordinates = (0, 0, 0, 0)

        self.rectangle_ROI_width = 0
        self.rectangle_ROI_height = 0
        self.rectangle_ROI_zoom = 1.0
        self.rectangle_ROI_zoom_count = 8
        self.j = 0

        self.get_image_dimensions()
        self.initialize_rectangle_ROI()

    def get_image_dimensions(self):
        (
            self.navigation_image_height,
            self.navigation_image_width,
            _,
        ) = self.navigation_image.shape
        self.navigation_image_aspect_ratio = (
            self.navigation_image_height / self.navigation_image_width
        )
        self.annotation_image_coordinates = (
            0,
            self.navigation_image_width,
            0,
            self.navigation_image_height,
        )

    def initialize_rectangle_ROI(self):
        self.update_rectangle_size()

    def update_rectangle_size(self):
        min_image_dimension = min(
            self.navigation_image_width,
            self.navigation_image_height,
        )
        min_rectangle_side = 10
        max_zoom_count = max(1.0, min_image_dimension / min_rectangle_side)
        self.rectangle_ROI_zoom_count = np.clip(
            self.rectangle_ROI_zoom_count,
            1.0,
            max_zoom_count,
        )

        rectangle_side = int(min_image_dimension / self.rectangle_ROI_zoom_count)
        rectangle_side = max(min_rectangle_side, rectangle_side)

        self.rectangle_ROI_zoom = self.rectangle_ROI_zoom_count
        self.rectangle_ROI_width = rectangle_side
        self.rectangle_ROI_height = rectangle_side

    def draw_rectangle_ROI(self, mouse_x, mouse_y, color):
        rectangle_color = color
        rectangle_width = 2

        # TODO: Better names, those are the rectangle ROI points!
        self.x1 = int(mouse_x - self.rectangle_ROI_width / 2)
        self.y1 = int(mouse_y - self.rectangle_ROI_height / 2)
        self.x2 = int(mouse_x + self.rectangle_ROI_width / 2)
        self.y2 = int(mouse_y + self.rectangle_ROI_height / 2)

        self.x1 = max(2, self.x1)
        self.y1 = max(2, self.y1)

        self.x2 = min(self.navigation_image_width - 2, self.x2)
        self.y2 = min(self.navigation_image_height - 2, self.y2)

        self.annotation_image_coordinates = (self.x1, self.x2, self.y1, self.y2)

        self.navigation_image = cv2.rectangle(
            self.navigation_image_original.copy(),
            (self.x1, self.y1),
            (self.x2, self.y2),
            rectangle_color,
            rectangle_width,
        )

        self.navigation_image_buffer = self.navigation_image.copy()

    def update_annotation_image(self):
        x1, x2, y1, y2 = self.annotation_image_coordinates
        image_ROI = self.navigation_image_original[y1:y2, x1:x2]

        new_width = (
            self.rectangle_ROI_zoom
            * (x2 - x1)
            * (self.config["image_viewer"]["width"] / self.navigation_image_width)
        )
        new_height = (
            self.rectangle_ROI_zoom
            * (y2 - y1)
            * (self.config["image_viewer"]["height"] / self.navigation_image_width)
        )

        new_size = (
            int(new_width),
            int(new_height),
        )

        self.annotation_image = cv2.resize(image_ROI, new_size)
        self.annotation_image_buffer = self.annotation_image.copy()
