import cv2
import numpy as np

COLOR_MAP = []
for c in range(93):
    c = (np.random.random((1, 3))*0.6+0.4).tolist()[0]
    COLOR_MAP.append(c)


def normalize_image(img):
    img = img.permute(1, 2, 0)
    img = img.cpu().data.numpy()
    img = (img - img.min()) / (img.max() - img.min())
    img = img.transpose(2, 0, 1) * 255
    return img


def draw_boxes(img, boxes):
    img = img.transpose(1, 2, 0)
    img = img.astype(np.uint8)
    img = img.copy()

    for box_elem in boxes:
        box, cat = box_elem
        color_elem = COLOR_MAP[cat]
        color = (
            int(color_elem[0]*255),
            int(color_elem[1]*255),
            int(color_elem[2]*255)
        )
        cv2.polylines(img, [box], True, color, 2)

    img = img.transpose(2, 0, 1)
    return img
