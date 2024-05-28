import cv2
import h5py
from pathlib import Path

def generate_txt(h5_path):
    # Reading the .h5 file
    h5_dataset = h5py.File(h5_path, 'r')

    # Get images
    image_files = h5_dataset.keys()

    for image_file in image_files:

        header = "image_name, ellipse_center_x, ellipse_center_y, bbox_height, bbox_width," \
                 " ellipse_main_axis, ellipse_secondary_axis, ellipse_angle"

        output = []

        file_name = f'img_{image_file}_contours.txt'

        original_img = h5_dataset[image_file]['img'][...]
        for contour_id in h5_dataset[image_file]['contours']:
            contour = h5_dataset[image_file]['contours'][contour_id]

            if len(contour[...]) >= 5:
                ((center_x, center_y), (ellipse_width, ellipse_height), ellipse_angle) = cv2.fitEllipse(contour[...])

                original_img = cv2.ellipse(original_img, (int(center_x), int(center_y)),
                                           (int(ellipse_width / 2), int(ellipse_height / 2)),
                                           ellipse_angle, 0, 360, (0, 0, 255), 2)
            else:
                break

            bbox_x, bbox_y, bbox_width, bbox_height = cv2.boundingRect(contour[...])
            original_img = cv2.rectangle(original_img, (bbox_x, bbox_y),
                                         (bbox_x + bbox_width, bbox_y + bbox_height), (0, 255, 0), 1)

            output.append([f"{image_file}, {center_x:.2f}, {center_y:.2f}, {bbox_height:.2f}, {bbox_width:.2f},"
                           f" {ellipse_width:.2f}, {ellipse_height:.2f}, {ellipse_angle:.2f}"])

        # cv2.imwrite(f'cnts_{image_file}', original_img)
        # this line above generate the images with the bounding boxes and the ellipses

        with open(file_name, 'w', encoding='utf-8') as arquivo:
            arquivo.write(header + '\n')
            for linha in output:
                arquivo.write(str(linha)[2:-2] + '\n')


# Path to .h5 file - Modify it accordingly
h5_file = Path(r"output/data.h5")

generate_txt(h5_file)
