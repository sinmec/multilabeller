import numpy as np

class Circle:
    def __init__(self, id, ROI):
        self.i = 0
        self.points = [None, None]
        self.translated_points = [None, None]
        self.color = (255, 0, 0)
        self.thickness = 3
        self.ROI = ROI

    def add_circle_points(self, point_x, point_y, translated_point_x, translated_point_y):
        if self.i <= 1:
            self.points[self.i] = (point_x, point_y)
            self.translated_points[self.i] = (translated_point_x, translated_point_y)
            if self.i < 1:
                self.i += 1
            else:
                self.i = 2
                self.create_circle()

    def display_updated_circle_zoomed_image(self):
        # circle on the annotation window

        self.center = [int((self.points[0][0] + self.points[1][0]) / 2),
                       int((self.points[0][1] + self.points[1][1]) / 2)]

        self.radius = int(np.sqrt(pow((self.points[1][0] - self.center[0]), 2) +
                                  pow((self.points[1][1] - self.center[1]), 2)))

    def create_circle(self):

            # circle on the annotation window

            self.center = [int((self.points[0][0] + self.points[1][0]) / 2),
                           int((self.points[0][1] + self.points[1][1]) / 2)]

            self.radius = int(np.sqrt(pow((self.points[1][0] - self.center[0]), 2) +
                                      pow((self.points[1][1] - self.center[1]), 2)))

            # circle on the navigation window

            self.translated_center = [int((self.translated_points[0][0] + self.translated_points[1][0]) / 2),
                                      int((self.translated_points[0][1] + self.translated_points[1][1]) / 2)]

            self.translated_circle_radius = int(np.sqrt(pow((self.translated_points[1][0] - self.translated_center[0]), 2) +
                                            pow((self.translated_points[1][1] - self.translated_center[1]), 2)))
            # todo: improve this
