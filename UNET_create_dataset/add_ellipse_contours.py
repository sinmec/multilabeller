import cv2

def add_ellipse_contours(event,x,y,flags,param):
   contour_collection = param[0]
   img = param[1]
   contour = contour_collection.contours[-1]
   if event == cv2.EVENT_LBUTTONDBLCLK and not contour.commit: # setting ellipse major points
       contour.d_1_points[contour.d_1_idx][0] = x
       contour.d_1_points[contour.d_1_idx][1] = y
       contour.d_1_idx = 1 - contour.d_1_idx
   elif event == cv2.EVENT_MBUTTONDOWN and not contour.commit: # commiting ellipse major points
       contour.d_1_points = tuple(contour.d_1_points)
       contour.commit = True
       contour.calculate_center()
       contour.calculate_d_1()
       contour.calculate_angle()
   elif contour.commit:
       contour.list_to_contour()
       if flags == cv2.EVENT_FLAG_SHIFTKEY + 7864322:
          contour.change_minor_axis_down()
       elif flags == cv2.EVENT_FLAG_SHIFTKEY + (-7864318):
          contour.change_minor_axis_up()
       if event == cv2.EVENT_MBUTTONDOWN: # commiting ellipse geometry
          contour_collection.draw_contour = True
