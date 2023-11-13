class Contour:
    def __init__(self, id):
        self.i = 0
        self.points = [None, None]
        self.translated_points = [None, None]
        self.contour_mode = False

    def contour_creation(self):
        self.contour_mode = not self.contour_mode

    def add_contour_points(self, point_x, point_y, translated_point_x, translated_point_y):
        if self.contour_mode:
            self.points[self.i] = (point_x, point_y)
            self.translated_points[self.i] = (translated_point_x, translated_point_y)
            if self.i < 1:
                self.i += 1
            else:
                self.i = 2
