class Circle:
    def __init__(self, image):
        self.i = 0
        self.zoomed_image = image
        self.points = [None, None]
        self.color = (0, 255, 0)
        self.thickness = 3

    def add_circle_points(self, point_x, point_y):
        if self.i <= 1:
            self.points[self.i] = (point_x, point_y)
            if self.i < 1:
                self.i += 1
            else:
                self.i = 2

