import cv2
import numpy as np
import yaml


class ImageManipulator:
    def __init__(self, image):
        self.config = None
        self.image_original = image.copy()
        self.zoomed_image_original = image.copy()

        self.image = image.copy()
        self.zoomed_image = image.copy()

        self.image_original_width = 0
        self.image_original_height = 0
        self.image_aspect_ratio = 0

        self.zoomed_image_coordinates = (0, 0)

        self.rectangle_ROI_width = 0
        self.rectangle_ROI_height = 0
        self.rectangle_ROI_zoom = 1.0
        self.rectangle_ROI_zoom_count = 1

        self.get_image_dimensions()
        self.initialize_rectangle_ROI()
        self.read_config_file()

    def get_image_dimensions(self):
        self.image_original_height, self.image_original_width, _ = self.image.shape
        self.image_aspect_ratio = self.image_original_height / self.image_original_width
        self.zoomed_image_coordinates = (0, self.image_original_width, 0, self.image_original_height)

    def initialize_rectangle_ROI(self):
        self.rectangle_ROI_width = self.image_original_width // self.rectangle_ROI_zoom_count
        self.rectangle_ROI_height = self.image_original_height // self.rectangle_ROI_zoom_count

    def update_rectangle_size(self):
        min_rectangle_width = 10
        min_rectangle_height = min_rectangle_width * self.image_aspect_ratio

        max_zoom = float(self.image_original_height / min_rectangle_height)
        min_zoom = 1.01

        h = 1.0 / max_zoom

        self.rectangle_ROI_zoom = np.clip(h * self.rectangle_ROI_zoom_count, min_zoom, max_zoom)

        self.rectangle_ROI_width = int(self.image_original_width / self.rectangle_ROI_zoom)
        self.rectangle_ROI_height = int(self.image_original_height / self.rectangle_ROI_zoom)

    def draw_rectangle_ROI(self, mouse_x, mouse_y):
        rectangle_color = (0, 255, 0)
        rectangle_width = 2

        # TODO: Better names, those are the rectangle ROI points!
        self.x1 = int(mouse_x - self.rectangle_ROI_width / 2)
        self.y1 = int(mouse_y - self.rectangle_ROI_width / 2)
        self.x2 = int(mouse_x + self.rectangle_ROI_width / 2)
        self.y2 = int(mouse_y + self.rectangle_ROI_width / 2)

        self.x1 = max(2, self.x1)
        self.y1 = max(2, self.y1)

        self.x2 = min(self.image_original_width - 2, self.x2)
        self.y2 = min(self.image_original_height + 2, self.y2)

        self.zoomed_image_coordinates = (self.x1, self.x2, self.y1, self.y2)

        self.image = cv2.rectangle(self.image_original.copy(), (self.x1, self.y1), (self.x2, self.y2), rectangle_color,
                                   rectangle_width)  # TODO: Why do we need this copy?
        self.image_rectangle_clean = self.image.copy()  # TODO:What is this _clean?

    def update_zoomed_image(self):
        x1, x2, y1, y2 = self.zoomed_image_coordinates
        image_ROI = self.image_original[y1:y2, x1:x2]

        # new_size = (int(self.rectangle_ROI_zoom * (x2 - x1)),
        #            int(self.rectangle_ROI_zoom * (y2 - y1)))

        new_size = (int((self.rectangle_ROI_zoom * (x2 - x1) * (500 / self.image_original_width))),
                    int((self.rectangle_ROI_zoom * (y2 - y1) * (500 / self.image_original_width))))

        self.zoomed_image = cv2.resize(image_ROI, new_size)
        self.zoomed_image_clean = self.zoomed_image  # TODO:What is this _clean?

    def draw_annotation_point(self, image, point_x, point_y):
        # TODO: This is dumb! Think on a smart solution!
        if (point_x is None) or (point_y is None):
            return
        circle_radius = 5
        circle_color = (0, 255, 0)
        circle_thickness = -1

        cv2.circle(image, (point_x, point_y), circle_radius, circle_color, circle_thickness)

    def read_config_file(self):
        try:
            with open("/home/sinmec/multilabeller/src/multilabeller/config.yml", "r") as config_file:
                # TODO: think on a better way to import config.yaml
                self.config = yaml.safe_load(config_file)
        except FileExistsError:
            print("Configuration file 'config.yml' was not found.")

    def translate_points(self, point_x, point_y):
        # TODO: This is dumb! Think on a smart solution!
        if (point_x is None) or (point_y is None):
            return None, None
        point_x_translated = self.x1 + int((point_x / self.rectangle_ROI_zoom)
                                           * (self.image_original_width / (self.config['image_viewer']['width'])))
        point_y_translated = self.y1 + int((point_y / self.rectangle_ROI_zoom)
                                           * (self.image_original_width / (self.config['image_viewer']['height'])))
        return point_x_translated, point_y_translated
