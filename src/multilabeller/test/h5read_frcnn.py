import cv2
import h5py
from pathlib import Path

# Path to .h5 file - Modify it accordingly
h5_file = Path(r"output/data.h5")

# Reading the .h5 file
h5_dataset = h5py.File(h5_file, 'r')

# Get images
image_files = h5_dataset.keys()

# formato do arquivo = ["nome_da_imagem OK", "centro_elipse_x", "centro_elipse_y", "altura_bbox OK",
#                       "comprimento_bbox OK", "eixo_principal_elipse", "eixo_secundario_elipse", "angulo_elipse"]

# nome do arquivo = {nome_da_image}_contours.txt

for image_file in image_files:
    original_img = h5_dataset[image_file]['img'][...]
    for contour_id in h5_dataset[image_file]['contours']:
        contour = h5_dataset[image_file]['contours'][contour_id]
        cv2.drawContours(original_img, [contour[...]], -1, [0, 0, 255], 2)
        x, y, w, h = cv2.boundingRect(contour[...])
        original_img = cv2.rectangle(original_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imwrite(f'cnts_{image_file}', original_img)

