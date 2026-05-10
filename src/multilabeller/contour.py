import numpy as np

RED = (255, 0, 0)
GREEN = (0, 0, 255)


class Contour:
    def __init__(self):
        self.index_points = 0
        self.points_annotation_window = []
        self.points_navigation_window = []
        self.points_image = []

        self.valid = True
        self.color = RED
        self.thickness = 3

        self.in_progress = True
        self.finished = False

        self.annotation_window_contour = None
        self.navigation_window_contour = None

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
        print("to_cv2_contour - Not implemented!")

    def add_points(self):
        print("add_points - Not implemented!")

    def annotation_point_to_image_point(self, point_annotation_window, target):
        x1, x2, y1, y2 = target.annotation_image_coordinates

        annotation_width = target.annotation_image.shape[1]
        annotation_height = target.annotation_image.shape[0]

        scale_x = (x2 - x1) / annotation_width
        scale_y = (y2 - y1) / annotation_height

        point_x = x1 + point_annotation_window[0] * scale_x
        point_y = y1 + point_annotation_window[1] * scale_y

        return [point_x, point_y]

    def image_point_to_annotation_point(self, point_image, target):
        x1, x2, y1, y2 = target.annotation_image_coordinates

        annotation_width = target.annotation_image.shape[1]
        annotation_height = target.annotation_image.shape[0]

        scale_x = annotation_width / (x2 - x1)
        scale_y = annotation_height / (y2 - y1)

        point_x = (point_image[0] - x1) * scale_x
        point_y = (point_image[1] - y1) * scale_y

        return [round(point_x), round(point_y)]

    def image_point_to_navigation_point(self, point_image, target):
        scale = getattr(target, "navigation_display_scale", 1.0)
        return [round(point_image[0] * scale), round(point_image[1] * scale)]

    def translate_from_annotation_to_image(self, target):
        self.points_image = [None for _ in range(len(self.points_annotation_window))]

        for i, point_annotation_window in enumerate(self.points_annotation_window):
            if point_annotation_window is None:
                continue

            self.points_image[i] = self.annotation_point_to_image_point(
                point_annotation_window,
                target,
            )

    def translate_from_image_to_annotation_window(self, target):
        if not self.points_image:
            self.points_annotation_window = []
            return

        x1, x2, y1, y2 = target.annotation_image_coordinates
        annotation_width = target.annotation_image.shape[1]
        annotation_height = target.annotation_image.shape[0]
        scale_x = annotation_width / (x2 - x1)
        scale_y = annotation_height / (y2 - y1)

        pts = np.array(self.points_image, dtype=float)
        ann_pts = np.empty_like(pts)
        ann_pts[:, 0] = (pts[:, 0] - x1) * scale_x
        ann_pts[:, 1] = (pts[:, 1] - y1) * scale_y
        self.points_annotation_window = ann_pts.tolist()

    def translate_from_image_to_navigation_window(self, target):
        if not self.points_image:
            self.points_navigation_window = []
            return

        scale = getattr(target, "navigation_display_scale", 1.0)
        pts = np.array(self.points_image, dtype=float)
        nav_pts = np.round(pts * scale).astype(int)
        self.points_navigation_window = nav_pts.tolist()

    def update_window_points_from_image_points(self, target):
        self.translate_from_image_to_annotation_window(target)
        self.translate_from_image_to_navigation_window(target)

    def translate_from_annotation_to_navigation_windows(self, target):
        self.translate_from_annotation_to_image(target)
        self.translate_from_image_to_navigation_window(target)

    def translate_from_navigation_to_annotation_windows(self, target):
        self.update_window_points_from_image_points(target)
