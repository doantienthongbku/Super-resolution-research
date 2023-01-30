# Accurate Image Super-Resolution Using Very Deep Convolutional Networks

This repository is implementation of the [Accurate Image Super-Resolution Using Very Deep Convolutional Networks](https://arxiv.org/abs/1511.04587)

## Abstract

We present a highly accurate single-image super-resolution (SR) method. Our method uses a very deep convolutional network inspired by VGG-net used for ImageNet classification \cite{simonyan2015very}. We find increasing our network depth shows a significant improvement in accuracy. Our final model uses 20 weight layers. By cascading small filters many times in a deep network structure, contextual information over large image regions is exploited in an efficient way. With very deep networks, however, convergence speed becomes a critical issue during training. We propose a simple yet effective training procedure. We learn residuals only and use extremely high learning rates (104 times higher than SRCNN \cite{dong2015image}) enabled by adjustable gradient clipping. Our proposed method performs better than existing methods in accuracy and visual improvements in our results are easily noticeable.

## Dataset

This dataset collect from [Lornatang on GitHub](https://github.com/Lornatang) contains T91, Set5, Set14, BSDS100 and BSDS200, etc.

- [Google Driver](https://drive.google.com/drive/folders/1A6lzGeQrFMxPqJehK9s37ce-tPDj20mD?usp=sharing)

## Training and Testing

You must prepare dataset by running run.py file on setup folder before training

After that, setup your own setting in config file then run the tools/train.py or tools/val.py to training or testing

## Result

### Set5

| Eval. Mat | Scale | SRCNN | SRCNN (reimplementation) |
|-----------|-------|-------|--------------|
| PSNR | 2 | 37.53 | 37.52 | 37.31
| PSNR | 3 | 33.66 | - |
| PSNR | 4 | 31.35 | - |

## Some tips

In this project I convert RGB to YCbCr color, it can be helpful in a super resolution problem, as it separates color information (chrominance) from luminance. This can be useful in image interpolation and upscaling, as it allows the algorithm to target the color information separately, which can help reduce artifacts or unwanted color shifts that can be caused by upscaling.

## Reference
[1] Kim, Jiwon, Jung Kwon Lee, and Kyoung Mu Lee. "Accurate image super-resolution using very deep convolutional networks." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.

[2] This implementaion repository: https://github.com/Lornatang/VDSR-PyTorch

