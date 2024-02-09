class Selector:
    def __init__(self):
        self.i = 0
        self.point_x = None
        self.point_y = None
        self.valid = False

    def update_point(self, point_x, point_y):
        self.point_x = point_x
        self.point_y = point_y
        self.valid = True

    def point_check(self):
        self.i += 1
