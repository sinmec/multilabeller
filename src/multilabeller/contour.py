class Contour:
    def __init__(self, id, ROI):
        self.i = 0
        self.points = []
        self.translated_points = []
        self.color = (255, 0, 0)
        self.thickness = 2
        self.ROI = ROI

    def add_contour_points(self, point, translated_point):
        if point is not None:
            self.points.append(point)
        if translated_point is not None:
            self.translated_points.append(translated_point)
        #print(self.points)
        #self.translated_points[self.i] = (translated_point_x, translated_point_y)
        self.i += 1
