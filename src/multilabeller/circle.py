class Circle:
    def __init__(self):
        self.i = 0
        self.points = [None, None]
        self.translated_points = [None, None]
        self.color = (255, 0, 0)
        self.thickness = 3

    def add_circle_points(self, point_x, point_y, translated_point_x, translated_point_y):
        if self.i <= 1:
            self.points[self.i] = (point_x, point_y)
            self.translated_points[self.i] = (translated_point_x, translated_point_y)
            if self.i < 1:
                self.i += 1
            else:
                self.i = 2
