import cv2
import numpy as np

from src.multilabeller.contour import Contour

RED = (255, 0, 0)
GREEN = (0, 0, 255)


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
        self.index_points = 0
        self.points_annotation_window = [None, None]
        self.points_navigation_window = []

        self.valid = True
        self.color = RED
        self.thickness = 3

        self.in_progress = True
        self.finished = False

        self.annotation_window_contour = None
        self.navigation_window_contour = None

        self.selected = False

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
