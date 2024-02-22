import cv2


def add_contours(event, x, y, flags, contour_collection):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        contour = contour_collection.contours[-1]
        contour.points_annotation_window.append((x, y))
        print("adding points to contour %03d" % contour.ID)
    elif event == cv2.EVENT_MBUTTONDOWN:
        contour = contour_collection.contours[-1]
        contour_collection.draw_contour = True
        print("commiting points from contour %03d" % contour.ID)
