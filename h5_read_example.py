import cv2
import h5py
from pathlib import Path

# Path to .h5 file - Modify it accordingly
h5_file = Path(r'C:\Users\rafaelfc\Downloads\dset_exp_N1200rpm_BEP1.2_D1000rpm.h5')

# Reading the .h5 file
h5_dataset = h5py.File(h5_file, 'r')

# Show attributes
print(h5_dataset.keys())

# There's a '1200_rpm_1.2_BEP_menosgotas_fixed' group. Let's check its contents
print(h5_dataset['1200_rpm_1.2_BEP_menosgotas_fixed'].keys())
# It has a few groups (['img_000001.jpg', 'img_000201.jpg', 'img_000401.jpg', 'img_000601.jpg', 'img_000801.jpg', 'img_001001.jpg', 'img_001201.jpg'])
# Again, let's check its contents.... For instance 'img_001001.jpg'

print(h5_dataset['1200_rpm_1.2_BEP_menosgotas_fixed']['img_000401.jpg'].keys())
# We have ['contours', 'mask_img', 'original_img']. Let's check it!

original_img = h5_dataset['1200_rpm_1.2_BEP_menosgotas_fixed']['img_000401.jpg']['original_img']
# It's a <HDF5 dataset "original_img": shape (1536, 1536), type "|u1">
# It seems that is a 1536 x 1536 image. To retrieve its values, we use the '[...]' operator

# Let's get the image and save it as 'test.jpg'
original_img = h5_dataset['1200_rpm_1.2_BEP_menosgotas_fixed']['img_000401.jpg']['original_img'][...]
original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR) # converting to BGR (color image)
cv2.imwrite('test.jpg',original_img)

# Now, let's check what's inside the contours group...
contours = h5_dataset['1200_rpm_1.2_BEP_menosgotas_fixed']['img_000401.jpg']['contours']
print(contours)
# It says that has 806 members... Let's loop over them

# That's the looping sequence here
for contour_name in contours:
    contour = h5_dataset['1200_rpm_1.2_BEP_menosgotas_fixed']['img_000401.jpg']['contours'][contour_name]
    print(contour) # This show that each contour is a shape (17, 1, 2), type "<i8">... By the size, this is a cv2 contour

    # Let's now draw the contours over the image
    cv2.drawContours(original_img, [contour[...]], -1, [0,0,255], 2)

# And now let's write
cv2.imwrite('test_with_contours.jpg',original_img)

