# DCGAN [1] implementation on MatConvNet (any MCN version is compatible)
Deep Convolutional Generative Adversarial Network (DCGAN) implementation on MatConvNet 

# Prerequisite: 
1. Download CelebA database at: https://drive.google.com/file/d/0B7EVK8r0v71pZjFTYXZWM3FlRnM/view?usp=sharing
2. Download Pretrained DCGAN at: https://drive.google.com/file/d/0B7EVK8r0v71pZjFTYXZWM3FlRnM/view?usp=sharing

# How to run
Run main_start_dcgan.m (set opts.idx_gpus = 1 in line 7 if GPUs are supported, otherwise opts.idx_gpus = 0)

'main_start_dcgan.m' will perform:
1) Installation of MatConvNet (ver 1.0-beta24)
2) Formation of a image db file fitting to MatConvNet
3) Testing a pre-trained DCGAN
4) traing a new DCGAN
*Note: all hyper parameters for training (e.g., learning rate) are identical to that in the paper [1]


# Result
![alt text](https://github.com/sunghbae/dcgan-matconvnet/demo.png)


[1] Radford, Alec, Luke Metz, and Soumith Chintala. "Unsupervised representation learning with deep convolutional generative adversarial networks." arXiv preprint arXiv:1511.06434 (2015).
