class Contour:
    def __init__(self, id):
        self.i = 0
        self.points = []
        self.translated_points = []
        self.color = (255, 0, 0)
        self.thickness = 2

    def add_contour_points(self, point, translated_point):
        self.points.append(point)
        self.translated_points.append(translated_point)
        print(self.points)
        #self.translated_points[self.i] = (translated_point_x, translated_point_y)
        self.i += 1