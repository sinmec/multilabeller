import numpy as np

class drawed_contour:
    def __init__(self):
        self.points = []
        self.cv2_contour = []
        self.ID = 0
        self.valid = False
        self.type = "draw"
    def list_to_contour(self):
        N_points = len(self.points)
        cv2_contour = np.zeros((N_points,1,2), dtype=int)
        for i, (x,y) in enumerate(self.points):
            cv2_contour[i,0,0] = self.points[i][0]
            cv2_contour[i,0,1] = self.points[i][1]
        self.cv2_contour = cv2_contour
        return cv2_contour
