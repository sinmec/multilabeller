import cv2
import numpy as np

RED = (255, 0, 0)
GREEN = (0, 0, 255)


class Circle:
    def __init__(self):
        self.index_points = 0
        self.points = [None, None]
        self.translated_points = [None, None]

        self.valid = True
        self.color = RED
        self.thickness = 3

        self.in_progress = True
        self.finished = False

        self.contour = None

        self.selected = False

    def toggle_selection(self):
        if self.selected:
            self.selected = False
        else:
            self.selected = True

    def toggle_color(self):
        if self.color == RED:
            self.color = GREEN
        elif self.color == GREEN:
            self.color = RED

    def to_cv2_contour(self):
        ellipse_poly = cv2.ellipse2Poly(
            (self.center[0], self.center[1]), (self.radius, self.radius), 0, 360, 1, 1
        )
        N_points = len(ellipse_poly)
        cv2_contour = np.zeros((N_points, 1, 2), dtype=int)
        for i, (x, y) in enumerate(ellipse_poly):
            cv2_contour[i, 0, 0] = ellipse_poly[i][0]
            cv2_contour[i, 0, 1] = ellipse_poly[i][1]
        self.contour = cv2_contour

    def add_circle_points(self, point_x, point_y):
        self.points[self.index_points] = (point_x, point_y)
        self.index_points += 1

        if self.index_points == 2:
            self.create_circle()
            self.index_points = 0
            self.in_progress = False
            self.finished = True
            self.to_cv2_contour()

    def create_circle(self):
        self.center = [
            int((self.points[0][0] + self.points[1][0]) / 2),
            int((self.points[0][1] + self.points[1][1]) / 2),
        ]

        self.radius = int(
            np.sqrt(
                pow((self.points[1][0] - self.center[0]), 2)
                + pow((self.points[1][1] - self.center[1]), 2)
            )
        )
