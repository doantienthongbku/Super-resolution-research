# Image Super-Resolution Using Deep Convolutional Networks

This repository is implementation of the [Image Super-Resolution Using Deep Convolutional Networks](https://arxiv.org/abs/1501.00092)

## Differences from the original

- Added the zero-padding
- Used the Adam instead of the SGD
- Removed the weights initialization

## Dataset

The 91-image, Set5 dataset converted to HDF5 can be downloaded from the links below.

| Dataset | Scale | Type | Link |
|---------|-------|------|------|
| 91-image | 2 | Train | [Download](https://www.dropbox.com/s/2hsah93sxgegsry/91-image_x2.h5?dl=0) |
| 91-image | 3 | Train | [Download](https://www.dropbox.com/s/curldmdf11iqakd/91-image_x3.h5?dl=0) |
| 91-image | 4 | Train | [Download](https://www.dropbox.com/s/22afykv4amfxeio/91-image_x4.h5?dl=0) |
| Set5 | 2 | Eval | [Download](https://www.dropbox.com/s/r8qs6tp395hgh8g/Set5_x2.h5?dl=0) |
| Set5 | 3 | Eval | [Download](https://www.dropbox.com/s/58ywjac4te3kbqq/Set5_x3.h5?dl=0) |
| Set5 | 4 | Eval | [Download](https://www.dropbox.com/s/0rz86yn3nnrodlb/Set5_x4.h5?dl=0) |

## Training and Testing

You can read code and script file in scripts folder in detail

## Result

### Set5

| Eval. Mat | Scale | SRCNN | SRCNN (Ours) |
|-----------|-------|-------|--------------|
| PSNR | 2 | 36.66 | 36.52 |
| PSNR | 3 | 32.75 | - |
| PSNR | 4 | 30.49 | - |