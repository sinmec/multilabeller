import cv2
import numpy as np

from src.multilabeller.contour import Contour


class Ellipse(Contour):
    def __init__(self):
        super().__init__()
        self.points_annotation_window = [None, None, None, None]

        self.major_axis = 0
        self.minor_axis = -1
        self.angle = 0
        self.x_c = 0
        self.y_c = 0
        self.in_configuration = False

        self.parameters = []

    def add_points(self, point_x, point_y, target):

        self.points_annotation_window[self.index_points] = [point_x, point_y]
        self.index_points += 1

        if self.index_points == 2:
            self.configure_ellipse_parameters()
            self.in_configuration = True
            self.index_points = 0

    def configure_ellipse_parameters(self):
        self.x_c, self.y_c = self.calculate_center(self.points_annotation_window)
        self.major_axis = self.calculate_major_axis(self.points_annotation_window)
        self.angle = self.calculate_angle(self.points_annotation_window)
        self.list_to_contour()
        if self.minor_axis == -1:
            self.minor_axis = self.major_axis

    def create_minor_axis_annotation_points(self):

        p_2_x = int(self.x_c - 1 * self.minor_axis * np.sin(np.deg2rad(self.angle)))
        p_2_y = int(self.y_c + 1 * self.minor_axis * np.cos(np.deg2rad(self.angle)))

        self.points_annotation_window[2] = [p_2_x, p_2_y]

        p_2_x = int(self.x_c + 1 * self.minor_axis * np.sin(np.deg2rad(self.angle)))
        p_2_y = int(self.y_c - 1 * self.minor_axis * np.cos(np.deg2rad(self.angle)))

        self.points_annotation_window[3] = [p_2_x, p_2_y]

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

        x_c, y_c = self.calculate_center(self.points_annotation_window)
        major_axis = self.calculate_major_axis(self.points_annotation_window)
        minor_axis = self.calculate_minor_axis(self.points_annotation_window)
        angle = self.calculate_angle(self.points_annotation_window)
        ellipse_poly = cv2.ellipse2Poly(
            (x_c, y_c), (major_axis, minor_axis), int(angle), 360, 1, 1
        )
        N_points = len(ellipse_poly)
        cv2_contour = np.zeros((N_points, 1, 2), dtype=int)
        for i, (x, y) in enumerate(ellipse_poly):
            cv2_contour[i, 0, 0] = ellipse_poly[i][0]
            cv2_contour[i, 0, 1] = ellipse_poly[i][1]
        self.annotation_window_contour = cv2_contour

        x_c, y_c = self.calculate_center(self.points_navigation_window)
        major_axis = self.calculate_major_axis(self.points_navigation_window)
        minor_axis = self.calculate_minor_axis(self.points_navigation_window)
        angle = self.calculate_angle(self.points_navigation_window)
        ellipse_poly = cv2.ellipse2Poly(
            (x_c, y_c), (major_axis, minor_axis), int(angle), 360, 1, 1
        )
        N_points = len(ellipse_poly)
        cv2_contour = np.zeros((N_points, 1, 2), dtype=int)
        for i, (x, y) in enumerate(ellipse_poly):
            cv2_contour[i, 0, 0] = ellipse_poly[i][0]
            cv2_contour[i, 0, 1] = ellipse_poly[i][1]
        self.navigation_window_contour = cv2_contour

    def calculate_center(self, points):
        x_1 = points[0][0]
        x_2 = points[1][0]
        y_1 = points[0][1]
        y_2 = points[1][1]
        x_c = (x_1 + x_2) / 2
        y_c = (y_1 + y_2) / 2
        return int(x_c), int(y_c)

    def calculate_major_axis(self, points):
        x_1 = points[0][0]
        x_2 = points[1][0]
        y_1 = points[0][1]
        y_2 = points[1][1]
        dx = x_2 - x_1
        dy = y_2 - y_1
        major_axis = 0.5 * np.sqrt(dx**2.0 + dy**2.0)
        return int(major_axis)

    def calculate_minor_axis(self, points):
        x_1 = points[2][0]
        x_2 = points[3][0]
        y_1 = points[2][1]
        y_2 = points[3][1]
        dx = x_2 - x_1
        dy = y_2 - y_1
        minor_axis = 0.5 * np.sqrt(dx**2.0 + dy**2.0)
        return int(minor_axis)

    def calculate_angle(self, points):
        x_1 = points[0][0]
        x_2 = points[1][0]
        y_1 = points[0][1]
        y_2 = points[1][1]
        dx = x_2 - x_1
        dy = y_2 - y_1
        angle = np.arctan2(dy, dx)
        return np.rad2deg(angle)

    def list_to_contour(self):
        ellipse_poly = cv2.ellipse2Poly(
            (self.x_c, self.y_c),
            (self.major_axis, self.minor_axis),
            int(self.angle),
            360,
            1,
            1,
        )

        N_points = len(ellipse_poly)
        cv2_contour = np.zeros((N_points, 1, 2), dtype=int)
        for i, (x, y) in enumerate(ellipse_poly):
            cv2_contour[i, 0, 0] = ellipse_poly[i][0]
            cv2_contour[i, 0, 1] = ellipse_poly[i][1]
        self.ellipse_contour = cv2_contour
