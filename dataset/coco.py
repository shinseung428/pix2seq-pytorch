import os
import cv2
import torch
import random
import numpy as np
import torch.utils.data as data
import torch.nn.functional as F
import pytorch_lightning as pl

import albumentations as A
from albumentations.augmentations.transforms import PadIfNeeded
from albumentations.augmentations.transforms import Normalize
from albumentations.augmentations.transforms import ColorJitter
from albumentations.augmentations.geometric.resize import LongestMaxSize
from torch.utils.data import DataLoader

from PIL import Image
from pycocotools.coco import COCO

from dataset.loader_utils import (
    normalize_box, box2ltrb, create_label,
    add_noise_labels, drop_classes
)


class CocoDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""""
    def __init__(
        self, root, json,
        transform=None,
        class_num=80,
        bin_size=256,
        input_size=512,
        max_targets=100,
        class_mask_prob=0.5,
        crop_ratio=0.7,
        crop_prob=0.3,
        h_flip_prob=0.5,
        min_area=0.005,
        min_visibility=0.1,
        phase="train"
    ):
        """Set the path for images, captions and vocabulary wrapper.

        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """""
        self.root = root
        self.coco = COCO(json)
        self.ids = list(self.coco.imgs.keys())

        if phase=="train":
            # global_scales = [280, 312, 344, 376, 408, 440, 472, 504]
            # small_scales = [200, 300, 400]
            global_scales = [640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992, 1024, 1056, 1088, 1120]
            small_scales = [600, 700, 800, 900, 1000]

            self.transform = A.Compose([
                # ColorJitter(
                #     brightness=0.5,
                #     contrast=0.5,
                #     saturation=0.5,
                #     hue=0.5
                # ),
                A.HorizontalFlip(p=0.5),
                A.OneOrOther(
                    A.Compose([
                        LongestMaxSize(max_size=global_scales)
                    ]
                    ),
                    A.Compose([
                        LongestMaxSize(max_size=small_scales),
                        PadIfNeeded(
                            min_height=small_scales[0],
                            min_width=small_scales[0],
                            border_mode=0,
                            position="top_left",
                            value=[127.5, 127.5, 127.5],
                            always_apply=True
                        ),
                        A.RandomCrop(
                            width=int(small_scales[0]*crop_ratio),
                            height=int(small_scales[0]*crop_ratio),
                            p=crop_prob
                        ),
                        LongestMaxSize(max_size=global_scales)
                    ]
                    )
                ),
                PadIfNeeded(
                    min_height=input_size,
                    min_width=input_size,
                    border_mode=0,
                    position="top_left",
                    value=[127.5, 127.5, 127.5],
                    always_apply=True
                ),
                Normalize(),
            ], bbox_params=A.BboxParams(
                format='coco',
                min_area=(input_size**2)*min_area,
                min_visibility=0.1,
                label_fields=['class_labels']
            ))
        else:
            self.transform = A.Compose([
                LongestMaxSize(max_size=input_size),
                PadIfNeeded(
                    min_height=input_size,
                    min_width=input_size,
                    border_mode=0,
                    position="top_left",
                    value=[127.5, 127.5, 127.5],
                    always_apply=True
                ),
                Normalize()
            ], bbox_params=A.BboxParams(
                format='coco',
                min_area=(input_size**2)*min_area,
                min_visibility=0.1,
                label_fields=['class_labels']
            ))

        self.class_mask_prob = class_mask_prob

        self.input_w = input_size
        self.input_h = input_size

        self.obj_token_len = 5 # [x1, y1, x2, y2, class]
        self.bin_size = bin_size
        self.max_targets = max_targets

        self.num_coco_classes = class_num

        # 3 are [<start>, <eos>, <noise_class>]
        self.num_classes = bin_size + self.num_coco_classes + 4

        self.start_class = bin_size + self.num_coco_classes
        self.eos_class = bin_size + self.num_coco_classes + 1
        self.na_class = bin_size + self.num_coco_classes + 2
        self.noise_class = bin_size + self.num_coco_classes + 3

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""""
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        # anns = coco.loadAnns(ann_ids)
        path = coco.loadImgs(img_id)[0]['file_name']

        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        image = np.array(image).astype(np.float32)

        # Random shuffling of annotations
        random.shuffle(ann_ids)

        bboxes = []
        classes = []
        for ann_id in ann_ids:
            ann_elem = coco.loadAnns(ann_id)[0]
            if ann_elem["iscrowd"]:
                continue

            category = ann_elem['category_id']
            bbox = ann_elem['bbox']

            if bbox[0] >= bbox[0] + bbox[2] or \
                    bbox[1] >= bbox[1] + bbox[3]:
                continue

            bboxes.append(bbox)
            classes.append(category)

        if self.transform is not None:
            transformed = self.transform(
                image=image,
                bboxes=bboxes,
                class_labels=classes
            )

            image = transformed["image"]
            bboxes = transformed["bboxes"]
            classes = transformed["class_labels"]

        image = torch.tensor(image).permute(2, 0, 1)

        targets = []
        ann_count = 0
        for bbox, category in zip(bboxes, classes):
            box = box2ltrb(bbox)
            label = create_label(
                image, box, category,
                self.bin_size, self.input_w, self.input_h
            )

            targets += label
            ann_count += 1

        target_seq = targets + [self.eos_class]
        input_seq = [self.start_class] + targets


        # Pad dummy labels
        # Dummy label could be either:
        # 1. augmented gt box with noise class token
        # 2. random box in the image
        if ann_count < self.max_targets:
            to_add = self.max_targets - ann_count
            input_seq, target_seq = add_noise_labels(
                input_seq, target_seq, to_add,
                self.na_class, self.eos_class,
                self.noise_class, self.bin_size
            )

        # randomly drop input class tokens
        input_mask = drop_classes(input_seq, self.max_targets, self.class_mask_prob)

        input_seq = torch.tensor(input_seq)
        target_seq = torch.tensor(target_seq)
        input_mask = torch.tensor(input_mask)

        return image, input_seq, target_seq, input_mask

    def __len__(self):
        return len(self.ids)


class CocoDataModule(pl.LightningDataModule):
    def __init__(self, args):
        data_args = args["data"]
        self.data_dir = data_args["root_dir"]
        self.crop_ratio = data_args["crop_ratio"]
        self.crop_prob = data_args["crop_prob"]
        self.h_flip_prob = data_args["horizontal_flip_prob"]
        self.obj_min_area = data_args["object_min_area"]
        self.obj_min_visibility = data_args["object_min_visibility"]
        self.num_workers = data_args["num_workers"]
        self.prefetch_factor = data_args["prefetch_factor"]

        train_args = args["train"]
        self.class_num = train_args["class_num"]
        self.batch_size = train_args["batch_size"]
        self.input_size = train_args["image_size"]
        self.bin_size = train_args["bin_size"]
        self.max_targets = train_args["max_targets"]
        self.mask_prob = train_args["class_mask_prob"]


    def setup(self, stage=None):
        img_dir = os.path.join(self.data_dir, "train2017")
        json_path = os.path.join(self.data_dir, "annotations/instances_train2017.json")
        self.coco_train = CocoDataset(
            img_dir, json_path,
            class_num=self.class_num,
            bin_size=self.bin_size,
            input_size=self.input_size,
            max_targets=self.max_targets,
            class_mask_prob=self.mask_prob,
            crop_ratio=self.crop_ratio,
            crop_prob=self.crop_prob,
            h_flip_prob=self.h_flip_prob,
            min_area=self.obj_min_area,
            min_visibility=self.obj_min_visibility,
            phase="train"
        )

        img_dir = os.path.join(self.data_dir, "val2017")
        json_path = os.path.join(self.data_dir, "annotations/instances_val2017.json")
        self.coco_val = CocoDataset(
            img_dir, json_path,
            class_num=self.class_num,
            bin_size=self.bin_size,
            input_size=self.input_size,
            max_targets=self.max_targets,
            class_mask_prob=self.mask_prob,
            phase="val"
        )

    def train_dataloader(self):
        return DataLoader(
            self.coco_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.coco_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor
        )

    def test_dataloader(self):
        pass

    def teardown(self, stage=None):
        pass



