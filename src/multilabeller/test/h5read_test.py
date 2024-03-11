import cv2
import h5py
from pathlib import Path

# Path to .h5 file - Modify it accordingly
h5_file = Path(r"output/2024_03_11_09_11_aiaiaia.h5")

# Reading the .h5 file
h5_dataset = h5py.File(h5_file, 'r')

# Show attributes
print(h5_dataset.keys())

# Get images
image_files = h5_dataset.keys()

for image_file in image_files:
    original_img = h5_dataset[image_file]['img'][...]

    for contour_id in h5_dataset[image_file]['contours']:
        contour = h5_dataset[image_file]['contours'][contour_id]
        cv2.drawContours(original_img, [contour[...]], -1, [0,0,255], 2)

    cv2.imwrite(f'cnts_{image_file}', original_img)
