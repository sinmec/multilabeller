RED = (255, 0, 0)
GREEN = (0, 0, 255)


class Contour:
    def __init__(self):
        self.index_points = 0
        self.points_annotation_window = []
        self.points_navigation_window = []

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
        print("Not implemented!")

    def translate_from_annotation_to_navigation_windows(self, target):
        for point_annotation_window in self.points_annotation_window:
            if point_annotation_window is None:
                continue

            point_x = point_annotation_window[0]
            point_y = point_annotation_window[1]

            points_navigation_window_x = target.x1 + int(
                (point_x / target.rectangle_ROI_zoom)
                * (
                    target.navigation_image_width
                    / (target.config["image_viewer"]["width"])
                )
            )

            # TODO: why widht here? shouldnt it be height on 'target.image_original_width'?
            #      same applies for the translator below!
            points_navigation_window_y = target.y1 + int(
                (point_y / target.rectangle_ROI_zoom)
                * (
                    target.navigation_image_width
                    / (target.config["image_viewer"]["height"])
                )
            )

            self.points_navigation_window.append(
                [points_navigation_window_x, points_navigation_window_y]
            )

    def translate_from_navigation_to_annotation_windows(self, target):
        for i, points_navigation_window in enumerate(self.points_navigation_window):
            if points_navigation_window is None:
                continue

            point_x = points_navigation_window[0]
            point_y = points_navigation_window[1]

            point_annotation_window_x = int(
                (point_x - target.x1)
                * (
                    target.config["image_viewer"]["width"]
                    / target.navigation_image_width
                )
                * target.rectangle_ROI_zoom
            )
            point_annotation_window_y = int(
                (point_y - target.y1)
                * (
                    target.config["image_viewer"]["height"]
                    / target.navigation_image_width
                )
                * target.rectangle_ROI_zoom
            )

            self.points_annotation_window[i][0] = int(point_annotation_window_x)
            self.points_annotation_window[i][1] = int(point_annotation_window_y)
