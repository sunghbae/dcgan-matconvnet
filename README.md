# DCGAN implementation on MatConvNet 
Deep Convolutional Generative Adversarial Network (DCGAN) [1] implementation on MatConvNet 
*Note: Any MCN version is compatible

## Prerequisite: 
1. Download CelebA database at: https://drive.google.com/file/d/0B7EVK8r0v71pZjFTYXZWM3FlRnM/view?usp=sharing
2. Download Pretrained DCGAN at: https://www.dropbox.com/s/fvasd83oqgse7xr/net_dcgan.mat?dl=0

## How to play
Run main_start_dcgan.m (set 'opts.idx_gpus = 1' in line 7 if GPUs are supported, otherwise 'opts.idx_gpus = 0')

'main_start_dcgan.m' will perform:
1. Installation of MatConvNet (ver 1.0-beta24)
2. Formation of an image DB file fitting to MatConvNet
3. Testing a pre-trained DCGAN
4. Training a new DCGAN
*Note: All network hyper parameters for training (e.g., learning rate) are identically set to [1]


## Result
![alt text](https://github.com/sunghbae/dcgan-matconvnet/blob/master/demo.png)

[1] Radford, Alec, Luke Metz, and Soumith Chintala. "Unsupervised representation learning with deep convolutional generative adversarial networks." arXiv preprint arXiv:1511.06434 (2015).(https://arxiv.org/abs/1511.06434) 
