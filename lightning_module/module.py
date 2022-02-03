import os
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from torch import nn
from torchvision import transforms
from torch.optim.lr_scheduler import LinearLR
from torch.optim.lr_scheduler import StepLR

from models.pix2seq import Pix2Seq

from lightning_module.module_utils import get_boxes
from lightning_module.img_utils import normalize_image, draw_boxes
from lightning_module.optimizer_utils import WarmupLinearDecayLR

class Pix2SeqModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()

        bin_size = args["train"]["bin_size"]
        class_num = args["train"]["class_num"]
        self.total_class = bin_size + class_num + 4
        self.ignore_index = bin_size + class_num + 2

        self.obj_class_num = class_num

        self.image_log_interval_step = args["train"]["image_log_interval_step"]

        self.warmup_epochs = args["train"]["warmup_epochs"]
        self.max_epochs = args["train"]["max_epochs"]

        self.optimizer_name = args["train"]["optimizer"]
        self.scheduler_name = args["train"]["scheduler"]
        self.learning_rate = args["train"]["learning_rate"]
        self.weight_decay = args["train"]["weight_decay"]
        self.step_size = args["train"]["scheduler_step_size"]
        self.scheduler_scale_size = args["train"]["scheduler_scale_size"]

        self.net = Pix2Seq(
            args["model"],
            pad_id=self.ignore_index,
            class_num=self.total_class
        )
        self.net.train()

        # na class token set as an ignore index
        # self.criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        self.criterion = nn.CrossEntropyLoss()

        self.bin_size = args["train"]["bin_size"]

    def log_image(self, img, input_seq, target_seq, output):
        tensorboard = self.logger.experiment
        img = normalize_image(img[0])
        gt_boxes = get_boxes(
            input_seq[0], img,
            self.bin_size, is_input=True,
            class_num=self.obj_class_num
        )
        gt_img = draw_boxes(img, gt_boxes)

        pred_boxes = get_boxes(
            output[0], img, self.bin_size,
            class_num=self.obj_class_num
        )
        pred_img = draw_boxes(img, pred_boxes)

        tensorboard.add_image("gt", gt_img, self.global_step)
        tensorboard.add_image("pred", pred_img, self.global_step)

    def _get_optimizer(self):
        param = {
            "params": filter(lambda p: p.requires_grad, self.net.parameters()),
            "lr": self.learning_rate,
            "weight_decay": self.weight_decay,
            "eps": 1e-08
        }

        if self.optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(**param)
        else:
            print("Invalid optimizer found!!")
            exit(0)

        return optimizer

    def configure_optimizers(self):
        optimizer = self._get_optimizer()

        if self.scheduler_name == "lambda":
            def lr_foo(epoch):
                if epoch < self.warmup_epochs:
                    # warm up lr
                    lr_scale = 0.1 ** (self.warmup_epochs - epoch)
                else:
                    lr_scale = self.scheduler_scale_size ** epoch

                return lr_scale

            scheduler = LambdaLR(
                optimizer,
                lr_lambda=lr_foo
            )
        elif self.scheduler_name == "step":
            scheduler = StepLR(
                optimizer,
                step_size=self.step_size
            )
        elif self.scheduler_name == "linear":
            scheduler = WarmupLinearDecayLR(
                optimizer,
                warmup_factor=0.01,
                warmup_iters=self.warmup_epochs,
                warmup_method="linear",
                end_epoch=self.max_epochs,
                final_lr_factor=0.01,
            )
        else:
            scheduler = None
            print("Scheduler name not found!!")

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        img, input_seq, target_seq, input_mask = batch

        img = img.cuda()
        input_seq = input_seq.cuda()
        target_seq = target_seq.cuda()

        output_seq = self.net(img, input_seq, input_mask)

        output_seq_flat = output_seq.reshape(-1, self.total_class)
        target_seq_flat = target_seq.reshape(-1)
        loss = self.criterion(
            output_seq_flat[target_seq_flat!=self.ignore_index],
            target_seq_flat[target_seq_flat!=self.ignore_index]
        )

        if self.global_step % self.image_log_interval_step == 0:
            output_seq = output_seq.permute(0, 2, 1)
            self.log_image(img, input_seq, target_seq, output_seq.permute(0, 2, 1))

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        img, input_seq, target_seq, input_mask = batch
        output = self.net(img, input_seq)

        output = output.reshape(-1, self.total_class)
        target_seq = target_seq.reshape(-1)
        val_loss = self.criterion(
            output[target_seq!=self.ignore_index],
            target_seq[target_seq!=self.ignore_index]
        )

        return val_loss

    def validation_epoch_end(self, outs):
        val_loss = torch.stack(outs).mean()
        self.log("val_loss", val_loss)
