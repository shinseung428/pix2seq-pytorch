import os
import fire
import torch

from utils import prepare_dirs
from utils import parse_arguments

from dataset.coco import CocoDataModule

from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from lightning_module.module import Pix2SeqModule

def train(config):
    args = parse_arguments(config)

    base_dir, _, _ = prepare_dirs(args)

    coco = CocoDataModule(args)
    module = Pix2SeqModule(args)

    if args["test"]["weight_path"]:
        resume_path = os.path.join(base_dir, args["train"]["resume"])
    else:
        print("Should provide a valid weight path!!")

    trainer = Trainer(gpus=torch.cuda.device_count())
    trainer.test(module, coco, ckpt_path=resume_path)


if __name__ == "__main__":
    fire.Fire(train)


