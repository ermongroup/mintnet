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
<img src="https://github.com/chenlin9/Fully-Convolutional-Normalizing-Flows/blob/release/samples/MNIST_samples.png" width="200">
<img src="https://github.com/chenlin9/Fully-Convolutional-Normalizing-Flows/blob/release/mint_images/CIFAR10_samples.png" width="200">
<img src="https://github.com/chenlin9/Fully-Convolutional-Normalizing-Flows/blob/release/samples/ImageNet_samples.png" width="200">
</p>


## Running the experiments
```bash
python main.py --runner [runner name] --config [config file]
```

Here `runner name` is one of the following:

- `DensityEstimationRunner`. Experiments on MintNet density estimation.
- `ClassificationRunner`. Experiments on MintNet classification.

and `config file` is the directory of some YAML file in `configs/`.


For example, if you want to train MintNet density estimation model on MNIST, just run

```bash
python main.py --runner DensityEstimationRunner --config mnist_density_config.yml
```
