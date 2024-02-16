import cv2
import numpy as np

from src.multilabeller.contour import Contour


class Ellipse(Contour):
    def __init__(self):
        super().__init__()
        self.points_annotation_window = [None, None, None, None, None]

        # self.center = [None, None]
        # self.major_axis = 0
        # self.minor_axis = -1
        # self.angle = 0

        self.parameters = []

    def add_points(self, point_x, point_y, target):

        self.points_annotation_window[self.index_points] = [point_x, point_y]
        self.index_points += 1

        if self.index_points == 5:
            self.from_points_to_ellipse()
            self.convert_ellipse_to_annotation_points()
            self.translate_from_annotation_to_navigation_windows(target)
            self.to_cv2_contour()
            self.index_points = 0
            self.in_progress = False
            self.finished = True

    def from_points_to_ellipse(self):

        N_points = len(self.points_annotation_window)
        ellipse_points = np.zeros((N_points, 1, 2), dtype=int)
        for i, (x, y) in enumerate(self.points_annotation_window):
            ellipse_points[i, 0, 0] = self.points_annotation_window[i][0]
            ellipse_points[i, 0, 1] = self.points_annotation_window[i][1]
        self.parameters = cv2.fitEllipse(ellipse_points)

    def convert_ellipse_to_annotation_points(self):
        center_x = int(self.parameters[0][0])
        center_y = int(self.parameters[0][1])
        major_axis = int(self.parameters[1][0] / 2)
        minor_axis = int(self.parameters[1][1] / 2)
        angle = int(self.parameters[2])

        ellipse_points = cv2.ellipse2Poly(
            (center_x, center_y), (major_axis, minor_axis), angle, 360, 1, 1
        )
        N_points = len(ellipse_points)
        self.points_annotation_window = [[None, None] for _ in range(N_points)]
        for i, (x, y) in enumerate(ellipse_points):
            self.points_annotation_window[i][0] = x
            self.points_annotation_window[i][1] = y

    def to_cv2_contour(self):
        N_points = len(self.points_annotation_window)
        cv2_contour = np.zeros((N_points, 1, 2), dtype=int)
        for i, (x, y) in enumerate(self.points_annotation_window):
            cv2_contour[i, 0, 0] = self.points_annotation_window[i][0]
            cv2_contour[i, 0, 1] = self.points_annotation_window[i][1]
        self.annotation_window_contour = cv2_contour

        N_points = len(self.points_navigation_window)
        cv2_contour = np.zeros((N_points, 1, 2), dtype=int)
        for i, (x, y) in enumerate(self.points_navigation_window):
            cv2_contour[i, 0, 0] = self.points_navigation_window[i][0]
            cv2_contour[i, 0, 1] = self.points_navigation_window[i][1]
        self.navigation_window_contour = cv2_contour
