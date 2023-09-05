import cv2
import numpy as np

from drawed_contour import drawed_contour

class circle_contour(drawed_contour):
    def __init__(self):
        drawed_contour.__init__(self)
        self.type = "circle"
        self.radius = 2
        self.c_x = 0
        self.c_y = 0
    def change_radius_up(self):
        self.radius += 1
    def change_radius_down(self):
        self.radius = max(self.radius - 1,0)
    def set_circle_center(self,x,y):
        self.c_x = x
        self.c_y = y
    def list_to_contour(self):
        ellipse_poly = cv2.ellipse2Poly((self.c_x, self.c_y), (self.radius, self.radius), 0, 360, 1, 1)
        N_points = len(ellipse_poly)
        print(N_points)
        cv2_contour = np.zeros((N_points,1,2), dtype=int)
        for i, (x,y) in enumerate(ellipse_poly):
            cv2_contour[i,0,0] = ellipse_poly[i][0]
            cv2_contour[i,0,1] = ellipse_poly[i][1]
        # print(cv2_contour)
        # exit()
        self.cv2_contour = cv2_contour
        return cv2_contour
