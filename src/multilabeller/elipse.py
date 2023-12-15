import numpy as np

class Elipse:
    def __init__(self, id):
        self.i = 0
        self.points = [None, None]
        self.translated_points = [None, None]
        self.color = (255, 0, 0)
        self.thickness = 3
        self.angle = 0
        self.y_axis = 2
        self.center = 0
        self.x_axis = 0

    def add_elipse_points(self, point_x, point_y, translated_point_x, translated_point_y):
        if self.i <= 1:
            self.points[self.i] = [point_x, point_y]
            print(self.points)
            self.translated_points[self.i] = [translated_point_x, translated_point_y]

    def create_initial_ellipse(self):
        # zoomed image

        x1 = self.points[0][0]
        x2 = self.points[1][0]
        y1 = self.points[0][1]
        y2 = self.points[1][1]

        self.center = (int((x1+x2)/2), int((y1+y2)/2))
        self.x_axis = int(np.sqrt(pow(x2-x1, 2) + pow(y2-y1, 2)))

        self.angle = int(np.degrees(np.arctan((y2 - y1) / (x2 - x1))))

        # manipulator image
        x1_translated = self.translated_points[0][0]
        x2_translated = self.translated_points[1][0]
        y1_translated = self.translated_points[0][1]
        y2_translated = self.translated_points[1][1]

        self.translated_center = (int((x1_translated + x2_translated) / 2), int((y1_translated + y2_translated) / 2))
        self.translated_x_axis = int(np.sqrt(pow(x2_translated - x1_translated, 2) + pow(y2_translated - y1_translated, 2)))

        self.translated_angle = int(np.degrees(np.arctan((y2_translated - y1_translated) / (x2_translated - x1_translated))))



    def define_y_axis(self, delta):
        if delta == 1:
            self.y_axis += 1
        elif delta == -1:
            self.y_axis -= 1

    def increase_ellipse_counter(self):
        self.i += 1