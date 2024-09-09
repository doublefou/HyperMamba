# HyperMamba
## Overview


## HFF Block


## Visual Inspection of HyperMamba


## Run
0. Requirements:
* python3
* pytorch 1.10
* torchvision 0.11.1
1. Training:
* Prepare the required images and store them in categories, set up training image folders and validation image folders respectively
* Run `python train.py`
2. Resume training:
* Modify `parser.add_argument('--RESUME', type=bool, default=True)` in `python train.py`
* Run `python train.py`
3. Testing:
* Run `python test.py`

## TensorBoard
Run `tensorboard --logdir runs --port 6006` to view training progress

## Reference
Some of the codes in this repo are borrowed from:  
* [Swin Transformer](https://github.com/microsoft/Swin-Transformer)  
* [ConvNeXt](https://github.com/facebookresearch/ConvNeXt)  
* [WZMIAOMIAO](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing)
* [HiFuse](https://github.com/huoxiangzuo/HiFuse)

## Citation

If you find our paper/code is helpful, please consider citing:

```bibtex

```

