import cv2
import matplotlib.pyplot as plt
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import numpy as np
import torch
import os


def show_outlines(maskgen_data):
    if len(maskgen_data) == 0:
        return

    # Define uma imagem sobreposta vazia onde serão desenhados os contornos
    ex_mask = maskgen_data[0]["segmentation"]
    img_h, img_w = ex_mask.shape[0:2]
    overlay_img = np.ones((img_h, img_w, 4))
    overlay_img[:, :, 3] = 0

    # gera contornos para cada segmentação gerada
    for each_gen in maskgen_data:

        # Gera os contornos baseado nas mascaras em formato boolean
        boolean_mask = each_gen["segmentation"]
        uint8_mask = 255 * np.uint8(boolean_mask)
        mask_contours, _ = cv2.findContours(uint8_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(mask_contours) == 0:
            continue

        # Desenha os contornos
        outline_opacity = 0.5
        outline_thickness = 2
        outline_color = np.concatenate([[0, 0, 1], [outline_opacity]])
        cv2.polylines(overlay_img, mask_contours, True, outline_color, outline_thickness, cv2.LINE_AA)

    # Desenha a imagem sobreposta
    ax = plt.gca()
    ax.set_autoscale_on(False)
    ax.imshow(overlay_img)

    return

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


def show_anns_individual_masks(anns, output_folder="/home/sinmec/multilabeller/src/multilabeller/test/data"):
    # # Cria uma pasta de saída se não existir
    os.makedirs(output_folder, exist_ok=True)

    # Itera sobre cada anotação
    for i, ann in enumerate(anns):
        # Cria uma imagem preta com quatro canais (RGBA)
        img = np.ones((ann['segmentation'].shape[0], ann['segmentation'].shape[1], 4))
        img[:, :, 3] = 0  # Define a transparência (canal alfa) para 0

        # Preenche os pixels da máscara com uma cor aleatória e transparência
        m = ann['segmentation']
        color_mask = np.concatenate(["white", [0.35]])
        img[m] = color_mask

        # Cria o caminho para o arquivo de imagem
        image_path = os.path.join(output_folder, f"mask_{i + 1}.png")

        # Exibe e salva a imagem
        plt.imshow(img)
        plt.axis('off')
        plt.savefig(image_path, bbox_inches="tight")
        cv2.waitKey(0)
        plt.clf()  # Limpa a figura atual para a próxima iteração

# lê a imagem
image = cv2.imread("/home/sinmec/multilabeller/src/multilabeller/SAM/ql_0_high_10_raw.png", 0)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# define qual modelo do sam sera usado. vit_b é o modelo mais leve porem o menos preciso
sam_checkpoint = "/home/sinmec/multilabeller/src/multilabeller/SAM/sam_vit_b_01ec64.pth"
model_type = "vit_b"

# Está configurado para rodar na gpu, a minha gpu só suporta rodar o modelo vit_b, caso contrario tenho que rodar na cpu
device = "cpu"

# configura o modelo do Sam
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

# gera a mascara
mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(image)

# plota a imagem de entrada com a mascara sobreposta, o show_anns mostra a mascara preenchida e o shw_outlines mostra apenas o contorno
plt.figure(figsize=(20,20))
plt.imshow(image)
# show_anns(masks)
show_outlines(masks)
plt.axis('off')
plt.show()

# # mostra cada mascara individualmente
# show_anns_individual_masks(masks)


a = 2