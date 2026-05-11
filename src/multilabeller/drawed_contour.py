import numpy as np

from src.multilabeller.contour import Contour


class DrawedContour(Contour):
    def __init__(self):
        super().__init__()

    def to_cv2_contour(self):
        self.annotation_window_contour = self._points_to_cv2_contour(
            self.points_annotation_window
        )
        self.navigation_window_contour = self._points_to_cv2_contour(
            self.points_navigation_window
        )

    def _points_to_cv2_contour(self, points):
        if len(points) == 0:
            return np.zeros((0, 1, 2), dtype=int)

        return np.asarray(points, dtype=int).reshape((-1, 1, 2))

    def add_points(self, point):
        if point is not None:
            self.points_annotation_window.append(point)
            self.index_points += 1
