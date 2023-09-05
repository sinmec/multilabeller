class drawed_contour_collection:
    def __init__(self):
        # self.contours = [drawed_contour()]
        self.contours = []
        self.draw_contour = False
        self.map_ID = []
        self.last_radius = 1
        self.last_d_2 = 2

    def add_contour(self, contour):
        ID = len(self.contours)
        contour.ID = ID
        self.contours.append(contour)
        if contour.type == "circle":
            contour.radius = self.last_radius

    def add_map_ID(self, map_ID):
        self.map_ID = map_ID
