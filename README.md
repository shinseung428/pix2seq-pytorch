# pix2seq-pytorch
Implementation of Pix2Seq [paper](https://arxiv.org/abs/2109.10852)


# Dataset
Download first [coco2017 dataset](https://cocodataset.org/#home) and put it under dataset folder.
```
- dataset
  - annotations
    - instances_train2017.json
    - instances_val2017.json
  - train2017
    - 000000000000.jpg
    - ...
  - val2017
```


# Train

### LOCAL
```
python train.py --config configs/pix2seq.yaml
```
