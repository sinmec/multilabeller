import numpy as np

from src.multilabeller.contour import Contour


class DrawedContour(Contour):
    def __init__(self):
        super().__init__()

    def to_cv2_contour(self):
        if self.points_annotation_window:
            pts = np.array(self.points_annotation_window, dtype=np.int32)
            self.annotation_window_contour = pts[:, np.newaxis, :]
        else:
            self.annotation_window_contour = np.zeros((0, 1, 2), dtype=np.int32)

        if self.points_navigation_window:
            pts = np.array(self.points_navigation_window, dtype=np.int32)
            self.navigation_window_contour = pts[:, np.newaxis, :]
        else:
            self.navigation_window_contour = np.zeros((0, 1, 2), dtype=np.int32)

    def add_points(self, point):
        if point is not None:
            self.points_annotation_window.append(point)
            self.index_points += 1
