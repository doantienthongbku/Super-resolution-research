# Accurate Image Super-Resolution Using Very Deep Convolutional Networks

This repository is implementation of the [Accurate Image Super-Resolution Using Very Deep Convolutional Networks](https://arxiv.org/abs/1511.04587)

## Abstract

We present a highly accurate single-image super-resolution (SR) method. Our method uses a very deep convolutional network inspired by VGG-net used for ImageNet classification \cite{simonyan2015very}. We find increasing our network depth shows a significant improvement in accuracy. Our final model uses 20 weight layers. By cascading small filters many times in a deep network structure, contextual information over large image regions is exploited in an efficient way. With very deep networks, however, convergence speed becomes a critical issue during training. We propose a simple yet effective training procedure. We learn residuals only and use extremely high learning rates (104 times higher than SRCNN \cite{dong2015image}) enabled by adjustable gradient clipping. Our proposed method performs better than existing methods in accuracy and visual improvements in our results are easily noticeable.

## Some tips

Converting RGB to YCbCr color can be helpful in a super resolution problem, as it separates color information (chrominance) from luminance. This can be useful in image interpolation and upscaling, as it allows the algorithm to target the color information separately, which can help reduce artifacts or unwanted color shifts that can be caused by upscaling.
