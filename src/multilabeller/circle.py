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
            self.translate_from_annotation_to_navigation_windows(target)
            self.index_points = 0
            self.in_progress = False
            self.finished = True


class WheelCircle(Contour):
    def __init__(self):
        super().__init__()
        self.points_annotation_window = [None]
        self.points_navigation_window = [None]
        self.radius_annotation_window = 10
        self.radius_navigation_window = 10
        self.in_configuration = False
        self.circle_contour = None

    def _create_cv2_contour(self, center, radius):
        ellipse_poly = cv2.ellipse2Poly(
            (center[0], center[1]), (radius, radius), 0, 360, 1, 1
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
        self.translate_from_annotation_to_navigation_windows(target)
        self.in_configuration = True

    def translate_from_annotation_to_navigation_windows(self, target):
        super().translate_from_annotation_to_navigation_windows(target)
        self.radius_navigation_window = int(
            self.radius_annotation_window
            / target.rectangle_ROI_zoom
            * (target.navigation_image_width / target.config["image_viewer"]["width"])
        )

    def translate_from_navigation_to_annotation_windows(self, target):
        super().translate_from_navigation_to_annotation_windows(target)
        self.radius_annotation_window = int(
            self.radius_navigation_window
            * (target.config["image_viewer"]["width"] / target.navigation_image_width)
            * target.rectangle_ROI_zoom
        )
