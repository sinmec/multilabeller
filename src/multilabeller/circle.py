class Circle:
    def __init__(self):
        self.i = 0
        self.points = [None, None]

    def add_circle_points(self, point_x, point_y):
        if self.i <= 1:
            self.points[self.i] = (point_x, point_y)
            if self.i < 1:
                self.i += 1
            else:
                self.i = 2
