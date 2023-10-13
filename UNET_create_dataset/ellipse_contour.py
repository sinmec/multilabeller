import cv2
import numpy as np

from drawed_contour import drawed_contour


class ellipse_contour(drawed_contour):
    def __init__(self):
        drawed_contour.__init__(self)
        self.type = "ellipse"
        self.d_1_points = [[0, 0], [0, 0]]
        self.d_1_idx = 0
        self.commit = False
        self.x_c = 0
        self.y_c = 0
        self.d_1 = 4
        self.d_2 = 4
        self.angle = 90

    def calculate_center(self):
        x_1 = self.d_1_points[0][0]
        x_2 = self.d_1_points[1][0]
        y_1 = self.d_1_points[0][1]
        y_2 = self.d_1_points[1][1]
        x_c = (x_1 + x_2) / 2
        y_c = (y_1 + y_2) / 2
        self.x_c = int(x_c)
        self.y_c = int(y_c)

    def calculate_d_1(self):
        x_1 = self.d_1_points[0][0]
        x_2 = self.d_1_points[1][0]
        y_1 = self.d_1_points[0][1]
        y_2 = self.d_1_points[1][1]
        dx = x_2 - x_1
        dy = y_2 - y_1
        d_1 = 0.5 * np.sqrt(dx**2.0 + dy**2.0)
        self.d_1 = int(d_1)

    def calculate_angle(self):
        x_1 = self.d_1_points[0][0]
        x_2 = self.d_1_points[1][0]
        y_1 = self.d_1_points[0][1]
        y_2 = self.d_1_points[1][1]
        dx = x_2 - x_1
        dy = y_2 - y_1
        angle = np.arctan2(dy, dx)
        angle = np.rad2deg(angle)
        self.angle = angle

    def change_minor_axis_up(self):
        self.d_2 += 1
        self.d_2 = min(self.d_2, self.d_1)

    def change_minor_axis_down(self):
        self.d_2 = max(self.d_2 - 1, 1)

    def list_to_contour(self):
        ellipse_poly = cv2.ellipse2Poly(
            (self.x_c, self.y_c), (self.d_1, self.d_2), int(self.angle), 360, 1, 1
        )
        N_points = len(ellipse_poly)
        print(N_points)
        cv2_contour = np.zeros((N_points, 1, 2), dtype=int)
        for i, (x, y) in enumerate(ellipse_poly):
            cv2_contour[i, 0, 0] = ellipse_poly[i][0]
            cv2_contour[i, 0, 1] = ellipse_poly[i][1]
        self.cv2_contour = cv2_contour
        return cv2_contour
