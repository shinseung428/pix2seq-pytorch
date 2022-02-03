import torch

from torch import nn
from models.backbone.resnet import resnet50
from models.encoder import Encoder
from models.decoder import Decoder
from models.pos_encoder import PositionalEncoder

from models.model_utils import init_weights

class Pix2Seq(nn.Module):
    def __init__(self, model_args, pad_id=346, class_num=349):
        super().__init__()
        self.backbone = resnet50(pretrained=True)
        # self.conv = nn.Conv2d(
        #     2048, model_args["enc_hidden_dim"],
        #     kernel_size=1
        # )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(2048, model_args["enc_hidden_dim"], kernel_size=1),
            nn.GroupNorm(32, model_args["enc_hidden_dim"])
        )

        max_len = model_args["max_w"] * model_args["max_h"] + 1
        self.encoder = Encoder(
            hidden_dim=model_args["enc_hidden_dim"],
            ff_dim=model_args["enc_ff_dim"],
            num_layers=model_args["enc_layers"],
            num_heads=model_args["enc_heads"],
            dropout=model_args["enc_dropout"],
            normalize_before=model_args["normalize_before"],
            activation=model_args["activation"]
        )
        self.enc_pos = PositionalEncoder(
            model_args["enc_hidden_dim"],
            max_len=max_len
        )

        self.decoder = Decoder(
            num_class=class_num,
            pad_id=pad_id,
            max_w=model_args["max_w"],
            max_h=model_args["max_h"],
            dec_max_len=model_args["dec_max_len"],
            hidden_dim=model_args["dec_hidden_dim"],
            ff_dim=model_args["dec_ff_dim"],
            num_layers=model_args["dec_layers"],
            num_heads=model_args["dec_heads"],
            dropout=model_args["dec_dropout"],
            normalize_before=model_args["normalize_before"],
            activation=model_args["activation"]
        )

        # nn.init.kaiming_normal_(
        #     self.bottleneck.weight,
        #     mode="fan_out",
        #     nonlinearity="relu"
        # )

        self.bottleneck.apply(init_weights)
        self.backbone.apply(init_weights)
        self.encoder.apply(init_weights)
        self.decoder.apply(init_weights)

        self.dec_max_len = model_args["dec_max_len"]

    def forward(self, x, input_seq, input_mask=None):
        x = self.backbone(x)
        x = self.bottleneck(x)

        # (b, c, h, w) -> (b, c, h*w) -> (h*w, b, c)
        x = x.flatten(2).permute(2, 0, 1)
        enc_pos = self.enc_pos(x)
        enc_out = self.encoder(x, pos=enc_pos)

        if input_seq.size(1) > 1:
            dec_out = self.decoder(
                enc_out, input_seq,
                src_pos=enc_pos,
                input_mask=input_mask
            )

            dec_out = dec_out.permute(1, 0, 2)

            return dec_out
        else:
            dummy_seq = input_seq
            for _ in range(self.dec_max_len):
                dec_out = self.decoder(
                    enc_out, dummy_seq
                )
                dec_out = dec_out.permute(1, 0, 2)

                pred = torch.softmax(dec_out, dim=-1)
                pred = torch.argmax(pred, dim=-1)
                pred = pred[:, -1].unsqueeze(-1)

                dummy_seq = torch.cat([dummy_seq, pred], dim=-1)

            return dummy_seq

