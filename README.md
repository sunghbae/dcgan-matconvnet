# DCGAN implementation on MatConvNet (any MCN version is compatible)
Deep Convolutional Generative Adversarial Network (DCGAN) implementation on MatConvNet 

# Prerequisite: 
1. Download CelebA database at: https://drive.google.com/file/d/0B7EVK8r0v71pZjFTYXZWM3FlRnM/view?usp=sharing
2. Download Pretrained DCGAN at: https://drive.google.com/file/d/0B7EVK8r0v71pZjFTYXZWM3FlRnM/view?usp=sharing

# How to run
Run main_start_dcgan.m (set opts.idx_gpus = 1 in line 7 if GPUs are supported, otherwise opts.idx_gpus = 0)
It then will do:
1) Install MatConvNet (ver 1.0-beta24)
2) Form a image db file fitting to MatConvNet
3) Test a pre-trained DCGAN
4) train a new DCGAN

# Result
![alt text](https://github.com/sunghbae/dcgan-matconvnet/demo.bmp)
