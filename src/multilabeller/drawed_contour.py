import cv2
import numpy as np

RED = (255, 0, 0)
GREEN = (0, 0, 255)


class DrawedContour:
    def __init__(self):
        self.index_points = 0
        self.points = []
        self.translated_points = []

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
        N_points = len(self.points)
        cv2_contour = np.zeros((N_points, 1, 2), dtype=int)
        for i, (x, y) in enumerate(self.points):
            cv2_contour[i, 0, 0] = self.points[i][0]
            cv2_contour[i, 0, 1] = self.points[i][1]
        self.contour = cv2_contour

    def add_contour_points(self, point):
        if point is not None:
            self.points.append(point)
            self.index_points += 1
