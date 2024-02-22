import numpy as np

from src.multilabeller.contour import Contour


class DrawedContour(Contour):
    def __init__(self):
        super().__init__()

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

    def add_points(self, point):
        if point is not None:
            self.points_annotation_window.append(point)
            self.index_points += 1
