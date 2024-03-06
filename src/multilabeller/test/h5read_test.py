import cv2
import h5py
from pathlib import Path

# Path to .h5 file - Modify it accordingly
h5_file = Path(r"C:\Users\higor\Documents\projetos\multilabeller\src\multilabeller\test\output\cnt_0.h5")

# Reading the .h5 file
h5_dataset = h5py.File(h5_file, 'r')

# Show attributes
print(h5_dataset.keys())

# Show contours attributes
print(h5_dataset['contours'].keys())

# Saving the original img
original_img = h5_dataset['img']['img'][...]
cv2.imwrite('original_image.jpg',original_img)

# Attributing the contours to the variable contours
contours = h5_dataset['contours']

for contour_name in contours:
    contour = h5_dataset['contours'][contour_name]
    print(contour)

    # Let's now draw the contours over the image
    cv2.drawContours(original_img, [contour[...]], -1, [0,0,255], 2)

cv2.imwrite('test_with_contours.jpg', original_img)
