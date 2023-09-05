import cv2

def add_contours(event,x,y,flags,contour_collection):
   # if event == cv2.EVENT_RBUTTONDOWN:
       # contour = drawed_contour()
       # contour_collection.add_contours(contour)
   if event == cv2.EVENT_LBUTTONDBLCLK:
       # cv2.circle(img,(x,y),1,(255,0,0),-1)
       contour = contour_collection.contours[-1]
       contour.points.append((x,y))
       print('adding points to contour %03d' % contour.ID)
   elif event == cv2.EVENT_MBUTTONDOWN:
       contour = contour_collection.contours[-1]
       # cv2_contour = contour.list_to_contour()
       contour_collection.draw_contour = True
       print('commiting points from contour %03d' % contour.ID)
