import cv2
import numpy as np

from src.multilabeller.contour import Contour


class Ellipse(Contour):
    def __init__(self):
        super().__init__()
        self.points_annotation_window = [None, None]

        self.center = [None, None]
        self.major_axis = 0
        self.minor_axis = -1
        self.angle = 0

        self.in_configuration = False

    def add_points(self, point_x, point_y, target):

        self.points_annotation_window[self.index_points] = [point_x, point_y]
        self.index_points += 1

        if self.index_points == 2:
            self.initialize_ellipse_shape()
            self.to_cv2_contour()
            self.index_points = 0
            self.in_progress = True
            self.in_configuration = True
    def initialize_ellipse_shape(self):
        x1 = self.points_annotation_window[0][0]
        x2 = self.points_annotation_window[1][0]
        y1 = self.points_annotation_window[0][1]
        y2 = self.points_annotation_window[1][1]

        self.center = [int((x1+x2)/2), int((y1+y2)/2)]
        self.major_axis = int(np.sqrt(pow(x2-x1, 2) + pow(y2-y1, 2)))
        if self.minor_axis == -1:
            self.minor_axis = self.major_axis
        self.angle = int(np.degrees(np.arctan((y2 - y1) / (x2 - x1))))



    def to_cv2_contour(self):

        #TODO: Fix zoom
        if not self.finished:
            ellipse_poly = cv2.ellipse2Poly(
                (self.center[0], self.center[1]), (self.major_axis//2, self.minor_axis//2), 0, 360, 1, 1
            )
        else:
            self.initialize_ellipse_shape()
            ellipse_poly = cv2.ellipse2Poly(
                (self.center[0], self.center[1]), (self.major_axis//2, self.minor_axis//2), 0, 360, 1, 1
            )
        N_points = len(ellipse_poly)

        cv2_contour = np.zeros((N_points, 1, 2), dtype=int)
        for i, (x, y) in enumerate(ellipse_poly):
            cv2_contour[i, 0, 0] = ellipse_poly[i][0]
            cv2_contour[i, 0, 1] = ellipse_poly[i][1]
        self.annotation_window_contour = cv2_contour

        #TODO: Fix zoom
        N_points = len(self.points_navigation_window)
        cv2_contour = np.zeros((N_points, 1, 2), dtype=int)
        for i, (x, y) in enumerate(self.points_navigation_window):
            cv2_contour[i, 0, 0] = self.points_navigation_window[i][0]
            cv2_contour[i, 0, 1] = self.points_navigation_window[i][1]
        self.navigation_window_contour = cv2_contour


