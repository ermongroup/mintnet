# MintNet: Building Invertible Neural Networks with Masked Convolutions
This repository contains the PyTorch implementation of our paper: 
[__MintNet: Building Invertible Neural Networks with Masked Convolutions__](https://arxiv.org/abs/1907.07945), _NeurIPS 2019_ .
We propose a new way of constructing invertible neural networks by combining simple building blocks with a novel set of composition rules. 
This leads to a rich set of invertible architectures, including those similar to 
ResNets. Inversion is achieved with a locally convergent iterative procedure 
that is parallelizable and very fast in practice. Additionally, 
the determinant of the Jacobian can be computed analytically and efficiently, 
enabling their generative use as flow models.

