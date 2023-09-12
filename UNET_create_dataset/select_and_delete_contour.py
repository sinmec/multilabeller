import cv2


def select_and_delete_contour(event, x, y, flags, contour_collection):
    ID = contour_collection.map_ID[y, x]
    if event == cv2.EVENT_LBUTTONDBLCLK and ID >= 0:
        contour_to_delete = contour_collection.contours[ID]
        contour_to_delete.valid = False
        cv2.drawContours(contour_collection.map_ID, [contour_to_delete.cv2_contour], -1, -1, -1)
        print("auto gener. contour %03d is deleted from the list" % ID)
