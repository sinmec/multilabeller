import cv2
import numpy as np

from src.multilabeller.contour import Contour


def create_circle(points):
    center = [
        int((points[0][0] + points[1][0]) / 2),
        int((points[0][1] + points[1][1]) / 2),
    ]

    radius = int(
        np.sqrt(pow((points[1][0] - center[0]), 2) + pow((points[1][1] - center[1]), 2))
    )

    return center, radius


class Circle(Contour):
    def __init__(self):
        super().__init__()
        self.points_annotation_window = [None, None]

    def to_cv2_contour(self):
        center, radius = create_circle(self.points_annotation_window)
        ellipse_poly = cv2.ellipse2Poly(
            (center[0], center[1]), (radius, radius), 0, 360, 1, 1
        )
        N_points = len(ellipse_poly)
        cv2_contour = np.zeros((N_points, 1, 2), dtype=int)
        for i, (x, y) in enumerate(ellipse_poly):
            cv2_contour[i, 0, 0] = ellipse_poly[i][0]
            cv2_contour[i, 0, 1] = ellipse_poly[i][1]
        self.annotation_window_contour = cv2_contour

        center, radius = create_circle(self.points_navigation_window)
        ellipse_poly = cv2.ellipse2Poly(
            (center[0], center[1]), (radius, radius), 0, 360, 1, 1
        )
        N_points = len(ellipse_poly)
        cv2_contour = np.zeros((N_points, 1, 2), dtype=int)
        for i, (x, y) in enumerate(ellipse_poly):
            cv2_contour[i, 0, 0] = ellipse_poly[i][0]
            cv2_contour[i, 0, 1] = ellipse_poly[i][1]
        self.navigation_window_contour = cv2_contour

    def add_points(self, point_x, point_y, target):
        self.points_annotation_window[self.index_points] = [point_x, point_y]
        self.index_points += 1

        if self.index_points == 2:
            self.translate_from_annotation_to_image(target)
            self.update_window_points_from_image_points(target)
            self.index_points = 0
            self.in_progress = False
            self.finished = True


class WheelCircle(Contour):
    def __init__(self):
        super().__init__()
        self.points_annotation_window = [None]
        self.points_navigation_window = [None]
        self.points_image = [None]
        self.radius_annotation_window = 10
        self.radius_navigation_window = 10
        self.radius_image = 10
        self.in_configuration = False
        self.circle_contour = None

    def _create_cv2_contour(self, center, radius):
        center_x = int(center[0])
        center_y = int(center[1])
        radius = int(radius)

        ellipse_poly = cv2.ellipse2Poly(
            (center_x, center_y), (radius, radius), 0, 360, 1, 1
        )
        N_points = len(ellipse_poly)
        cv2_contour = np.zeros((N_points, 1, 2), dtype=int)

        for i, (x, y) in enumerate(ellipse_poly):
            cv2_contour[i, 0, 0] = x
            cv2_contour[i, 0, 1] = y

        return cv2_contour

    def to_cv2_contour(self):
        if self.points_annotation_window[0] is not None:
            self.annotation_window_contour = self._create_cv2_contour(
                self.points_annotation_window[0],
                self.radius_annotation_window,
            )

        if self.points_navigation_window[0] is not None:
            self.navigation_window_contour = self._create_cv2_contour(
                self.points_navigation_window[0],
                self.radius_navigation_window,
            )

    def configure_circle_parameters(self):
        if self.points_annotation_window[0] is None:
            return

        self.circle_contour = self._create_cv2_contour(
            self.points_annotation_window[0],
            self.radius_annotation_window,
        )

    def add_points(self, point_x, point_y, target):
        self.points_annotation_window[0] = [point_x, point_y]
        self.translate_from_annotation_to_image(target)
        self.update_window_points_from_image_points(target)
        self.in_configuration = True

    def translate_from_annotation_to_image(self, target):
        super().translate_from_annotation_to_image(target)

        x1, x2, y1, y2 = target.annotation_image_coordinates

        annotation_width = target.annotation_image.shape[1]
        annotation_height = target.annotation_image.shape[0]

        scale_x = (x2 - x1) / annotation_width
        scale_y = (y2 - y1) / annotation_height

        radius_scale = (scale_x + scale_y) / 2

        self.radius_image = self.radius_annotation_window * radius_scale

    def translate_from_image_to_annotation_window(self, target):
        super().translate_from_image_to_annotation_window(target)

        x1, x2, y1, y2 = target.annotation_image_coordinates

        annotation_width = target.annotation_image.shape[1]
        annotation_height = target.annotation_image.shape[0]

        scale_x = annotation_width / (x2 - x1)
        scale_y = annotation_height / (y2 - y1)

        radius_scale = (scale_x + scale_y) / 2

        self.radius_annotation_window = self.radius_image * radius_scale

    def translate_from_image_to_navigation_window(self, target):
        super().translate_from_image_to_navigation_window(target)

        self.radius_navigation_window = self.radius_image
