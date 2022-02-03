import numpy as np


def normalize_box(
    box, img_width, img_height,
    target_width, target_height
):
    x, y, width, height = box
    x = x / img_width * target_width
    y = y / img_height * target_height

    width = width / img_width * target_width
    height = height / img_height * target_height

    box = [x, y, width, height]
    return box


def box2ltrb(box):
    x, y, width, height = box
    box = [x, y, x + width, y + height]

    return box


def create_label(
    image, box, category,
    bin_size, input_w, input_h
):
    assert input_w % bin_size == 0
    assert input_h % bin_size == 0

    x1, y1, x2, y2 = box

    x1_bin = np.round(x1 / input_w * (bin_size - 1)).astype(np.int32)
    y1_bin = np.round(y1 / input_h * (bin_size - 1)).astype(np.int32)
    x2_bin = np.round(x2 / input_w * (bin_size - 1)).astype(np.int32)
    y2_bin = np.round(y2 / input_h * (bin_size - 1)).astype(np.int32)

    category = category + bin_size - 1

    label = [x1_bin, y1_bin, x2_bin, y2_bin, category]

    return label


def sample_noisy_box(bin_size, min_size_ratio=0.02):
    # add a random box in the image region
    min_bin_size = int(bin_size * min_size_ratio)
    x1 = np.random.randint(0, bin_size - min_bin_size)
    y1 = np.random.randint(0, bin_size - min_bin_size)

    start_pt_x = x1 + min_bin_size
    start_pt_y = y1 + min_bin_size
    x2 = np.random.randint(start_pt_x, bin_size)
    y2 = np.random.randint(start_pt_y, bin_size)

    return [x1, y1, x2, y2]


def augment_gt_box(input_seq):
    input_seq = input_seq[1:]
    box_idx = np.random.randint(len(input_seq)/5)
    sampled_box = np.array(input_seq[box_idx*5:box_idx*5+4])

    width = sampled_box[2] - sampled_box[0]
    height = sampled_box[3] - sampled_box[1]
    min_size = max(min(width, height) // 5, 1)
    perturb_val = [
        np.random.randint(-min_size, min_size) for _ in range(4)
    ]

    sampled_box += np.array(perturb_val)
    x1, y1, x2, y2 = sampled_box

    return [x1, y1, x2, y2]

def is_box_valid(bbox, bin_size):
    x1, y1, x2, y2 = bbox

    is_valid = True
    if x1 >= x2 or y1 >= y2:
        is_valid = False

    bin_size -= 1
    if x1 < 0 or y1 < 0 or x2 > bin_size or y2 > bin_size:
        is_valid = False

    return is_valid

def add_noise_labels(
    input_seq, target_seq, to_add,
    na_class, eos_class,
    noise_class, bin_size
):
    dummy_counter = 0
    dummy_inputs = []
    dummy_targets = []
    processed_boxes = []
    while dummy_counter < to_add:
        if len(input_seq[1:]) > 0:
            if np.random.random() > 0.5:
                bbox = sample_noisy_box(bin_size)
            else:
                bbox = augment_gt_box(input_seq)
        else:
            bbox = sample_noisy_box(bin_size)

        if not is_box_valid(bbox, bin_size):
            continue

        if bbox in processed_boxes:
            continue
        else:
            processed_boxes.append(bbox)

        dummy_inputs += bbox + [noise_class]
        dummy_targets += [na_class, na_class, na_class, noise_class, eos_class]

        dummy_counter += 1

    input_seq += dummy_inputs
    target_seq += dummy_targets

    return input_seq, target_seq


def drop_classes(sequence, annotations, class_mask_prob):
    mask = [False] * (len(sequence) - 1)
    for idx in range(0, annotations-1):
        if np.random.random() < class_mask_prob:
            mask[idx*5+4] = True
    mask = [False] + mask
    return mask
