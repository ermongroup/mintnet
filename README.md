# MintNet: Building Invertible Neural Networks with Masked Convolutions
This repository contains the PyTorch implementation of our paper: 
[__MintNet: Building Invertible Neural Networks with Masked Convolutions__](https://arxiv.org/abs/1907.07945), _NeurIPS 2019_ .
We propose a new way of constructing invertible neural networks by combining simple building blocks with a novel set of composition rules. 
This leads to a rich set of invertible architectures, including those similar to 
ResNets. Inversion is achieved with a locally convergent iterative procedure 
that is parallelizable and very fast in practice. Additionally, 
the determinant of the Jacobian can be computed analytically and efficiently, 
enabling their generative use as flow models.


<p align="center">
<img src="https://github.com/pclucas14/pixel-cnn-pp/blob/master/images/pcnn_lr:0.00020_nr-resnet5_nr-filters160_143.png">
<img src="https://github.com/pclucas14/pixel-cnn-pp/blob/master/images/pcnn_lr:0.00020_nr-resnet5_nr-filters160_122.png">
<img src="https://github.com/pclucas14/pixel-cnn-pp/blob/master/images/pcnn_lr:0.00020_nr-resnet5_nr-filters160_137.png">
<img src="https://github.com/pclucas14/pixel-cnn-pp/blob/master/images/pcnn_lr:0.00020_nr-resnet5_nr-filters160_101.png">
</p>
