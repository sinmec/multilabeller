import cv2


def add_small_contours(event, x, y, flags, param):
    # print(key)
    contour_collection = param[0]
    img = param[1]
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(img, (x, y), 1, (255, 0, 0), -1)
        contour = contour_collection.contours[-1]
        contour.set_circle_center(x, y)
        # contour.points.append((x,y))
        print("setting contour center to contour %03d" % contour.ID)
    elif event == cv2.EVENT_MBUTTONDOWN:
        contour = contour_collection.contours[-1]
        cv2_contour = contour.list_to_contour()
        contour_collection.draw_contour = True
        print("commiting circle and creating contour %03d" % contour.ID)
    # if flags == cv2.EVENT_FLAG_SHIFTKEY and event== cv2.EVENT_MOUSEWHEEL:
    if flags == cv2.EVENT_FLAG_SHIFTKEY + 7864322:
        contour = contour_collection.contours[-1]
        cv2_contour = contour.list_to_contour()
        contour.change_radius_down()
    elif flags == cv2.EVENT_FLAG_SHIFTKEY + (-7864318):
        contour = contour_collection.contours[-1]
        cv2_contour = contour.list_to_contour()
        contour.change_radius_up()
