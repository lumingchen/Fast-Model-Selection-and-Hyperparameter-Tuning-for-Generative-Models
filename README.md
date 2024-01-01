# Adaptive Successive Halving

This is an implementation of AdaptSH and Successive Halving (SH) for GAN models.

## Installation
- Python==3.8.8
- Pytorch==1.8.0
- numpy
- scipy
- tqdm
- scikit-learn
- joblib
- cupy
- statsmodels


## Scripts

-   main.py : run this file to conduct experiments.
-   simpleGAN.py : neural network architectures and optimization objective for training GANs.
-   mmd.py : functions for calculating MMD and performing MMD based tests
-   tuning.py: contains implementations of adaptSH and SH
-   utils.py  : contains implementations of different sliced-based Wasserstein distances.
-   generate.py: functions to generate train and test data

## Examples

To perform model selection using AdaptSH on the half moons dataset:

```
python main.py --seed 2 --dataset MOON --lr 0.001 --B 1000 --train-size 1000 --test-size 500
```
To perform model selection using SH on the swiss roll dataset:
```
python main.py --seed 2 --dataset SWISSROLL --B 1000 --train-size 2000 --test-size 1000 --algorithm SH
```

