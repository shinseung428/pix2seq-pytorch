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

    base_dir, log_dir, save_dir = prepare_dirs(args)
    tb_logger = pl_loggers.TensorBoardLogger(log_dir)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            dirpath=save_dir,
            filename="pix2seq-{epoch:04d}-{val_loss:.4f}",
            save_top_k=100,
            mode="min",
    )

    if args["train"]["resume"]:
        resume_path = os.path.join(base_dir, args["train"]["resume"])
    else:
        resume_path = None

    coco = CocoDataModule(args)
    if args["train"]["weight_only"] and resume_path is not None:
        module = Pix2SeqModule.load_from_checkpoint(
            resume_path, args=args
        )
    else:
        module = Pix2SeqModule(args)


    trainer = Trainer(
        precision=16 if args["train"]["use_fp16"] else 32,
        max_epochs=args["train"]["max_epochs"],
        logger=tb_logger,
        flush_logs_every_n_steps=args["train"]["scalar_log_interval_step"],
        num_sanity_val_steps=0,
        check_val_every_n_epoch=1,
        callbacks=[lr_monitor, checkpoint_callback],
        gpus=torch.cuda.device_count(),
        sync_batchnorm=True,
        accelerator="ddp"
    )
    if args["train"]["weight_only"]:
        trainer.fit(module, coco)
    else:
        trainer.fit(module, coco, ckpt_path=resume_path)


if __name__ == "__main__":
    fire.Fire(train)


