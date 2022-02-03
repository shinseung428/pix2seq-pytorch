import numpy as np

import torch
import torch.nn.functional as F


def get_boxes(sequence, img, bin_size, is_input=False, class_num=80):
    img_h, img_w = img.shape[1:]
    if is_input:
        sequence = sequence[1:]
    else:
        sequence = sequence[:-1]
        sequence = F.softmax(sequence, dim=1)

        # change this later to nucleus sampling
        sequence = torch.argmax(sequence, dim=1)

    sequence = sequence.cpu().data.numpy()

    boxes = []
    for idx in range(0, len(sequence), 5):
        sliced_seq = sequence[idx:idx+5]
        x1, y1, x2, y2, cat = sliced_seq

        x1 = x1 / (bin_size - 1) * img_w
        y1 = y1 / (bin_size - 1) * img_h

        x2 = x2 / (bin_size - 1) * img_w
        y2 = y2 / (bin_size - 1) * img_h

        # skip of box coordinate is invalid
        if x1 > x2 or y1 > y2:
            continue

        bbox = np.array([
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2]
        ]).astype(np.int32)
        cat = cat - bin_size + 1
        if cat < 1 or cat > class_num:
            continue
        # if cat < 1 or cat > class_num:
        #     cat = 91
        boxes.append((bbox, cat))

    return boxes
