import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import math
from .utils import *


class ActNorm(nn.Module):
    """ An implementation of a activation normalization layer
    from Glow: Generative Flow with Invertible 1x1 Convolutions
    (https://arxiv.org/abs/1807.03039).
    """

    def __init__(self, config, num_inputs):
        super(ActNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(num_inputs)).to(config.device)
        self.bias = nn.Parameter(torch.zeros(num_inputs)).to(config.device)
        self.initialized = False

    def forward(self, inputs):
        if self.initialized == False:
            self.weight.data.copy_(1 / (inputs.std(0) + 1e-12))
            self.bias.data.copy_(inputs.mean(0))
            self.initialized = True
        return (inputs - self.bias) * self.weight


# DO NOT FORGET ACTNORM!!!
class BasicBlock(nn.Module):
    # Input_dim should be 1(grey scale image) or 3(RGB image), or other dimension if use SpaceToDepth

    def init_conv_weight(self, weight):
        # init.kaiming_uniform_(weight, math.sqrt(5.))
        # init.kaiming_normal_(weight, math.sqrt(5.))
        init.xavier_normal_(weight, 0.1)

    def init_conv_bias(self, weight, bias):
        fan_in, _ = init._calculate_fan_in_and_fan_out(weight)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(bias, -bound, bound)

    def __init__(self, config, shape, latent_dim, type, input_dim=3, kernel1=3, kernel2=3, kernel3=3, init_zero=False):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        # self.kernel = kernel
        self.padding1 = kernel1 // 2
        self.padding2 = kernel2 // 2
        self.padding3 = kernel3 // 2

        self.weight1 = nn.Parameter(
            torch.randn(input_dim * latent_dim, input_dim, kernel1, kernel1) * 1e-5
        )
        self.bias1 = nn.Parameter(
            torch.zeros(input_dim * latent_dim)
        )

        self.init_conv_weight(self.weight1)
        self.init_conv_bias(self.weight1, self.bias1)

        self.weight2 = nn.Parameter(
            torch.randn(input_dim * latent_dim, input_dim * latent_dim, kernel2, kernel2) * 1e-5
            # torch.zeros(input_dim * latent_dim, input_dim * latent_dim, kernel2, kernel2)
        )
        self.bias2 = nn.Parameter(
            torch.zeros(input_dim * latent_dim)
        )

        if not init_zero:
            self.init_conv_weight(self.weight2)
            self.init_conv_bias(self.weight2, self.bias2)

        self.weight3 = nn.Parameter(
            # torch.zeros(input_dim, input_dim * latent_dim, kernel3, kernel3)
            torch.randn(input_dim, input_dim * latent_dim, kernel3, kernel3) * 1e-5
        )

        self.bias3 = nn.Parameter(
            torch.zeros(input_dim)
        )

        self.init_conv_weight(self.weight3)
        self.init_conv_bias(self.weight3, self.bias3)

        # Define masks

        # Mask out the element above diagonal
        self.mask1 = nn.Parameter(torch.ones_like(self.weight1), requires_grad=False)
        self.center_mask1 = nn.Parameter(torch.zeros_like(self.weight1), requires_grad=False)
        self.mask2 = nn.Parameter(torch.ones_like(self.weight2), requires_grad=False)
        self.center_mask2 = nn.Parameter(torch.zeros_like(self.weight2), requires_grad=False)
        self.mask3 = nn.Parameter(torch.ones_like(self.weight3), requires_grad=False)
        self.center_mask3 = nn.Parameter(torch.zeros_like(self.weight3), requires_grad=False)

        for i in range(latent_dim):
            fill_mask(self.mask1[i * input_dim: (i + 1) * input_dim, ...], type=type)
            fill_center_mask(self.center_mask1[i * input_dim: (i + 1) * input_dim, ...])
            fill_mask(self.mask3[:, i * input_dim: (i + 1) * input_dim, ...], type=type)
            fill_center_mask(self.center_mask3[:, i * input_dim: (i + 1) * input_dim, ...])
            for j in range(latent_dim):
                fill_mask(self.mask2[i * input_dim: (i + 1) * input_dim, j * input_dim: (j + 1) * input_dim, ...],
                          type=type)
                fill_center_mask(
                    self.center_mask2[i * input_dim: (i + 1) * input_dim, j * input_dim: (j + 1) * input_dim, ...])

        self.non_linearity = F.elu

        self.t = nn.Parameter(torch.ones(1, *shape))

    def check_nan(self, inputs):
        if torch.isnan(inputs).any():
            import pdb
            pdb.set_trace()
            a = 1

    def forward(self, x):
        masked_weight1 = (self.weight1 * (1. - self.center_mask1) + torch.abs(
            self.weight1) * self.center_mask1) * self.mask1
        # shape: B x latent_output . input_dim x img_size x img_size
        latent_output = F.conv2d(x, masked_weight1, bias=self.bias1, padding=self.padding1, stride=1)
        latent_output = self.non_linearity(latent_output)

        masked_weight2 = (self.weight2 * (1. - self.center_mask2) + torch.abs(
            self.weight2) * self.center_mask2) * self.mask2

        latent_output = F.conv2d(latent_output, masked_weight2, bias=self.bias2, padding=self.padding2, stride=1)
        latent_output = self.non_linearity(latent_output)

        masked_weight3 = (self.weight3 * (1. - self.center_mask3) + torch.abs(
            self.weight3) * self.center_mask3) * self.mask3

        latent_output = F.conv2d(latent_output, masked_weight3, bias=self.bias3, padding=self.padding3, stride=1)

        t = torch.max(torch.abs(self.t), torch.tensor(1e-12, device=x.device))
        output = latent_output + t * x

        return output


class SpaceToDepth(nn.Module):
    def __init__(self, block_size):
        super(SpaceToDepth, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size * block_size

    def forward(self, x):
        output = x.permute(0, 2, 3, 1)
        (batch_size, s_height, s_width, s_depth) = output.size()
        d_depth = s_depth * self.block_size_sq
        d_width = int(s_width / self.block_size)
        d_height = int(s_height / self.block_size)
        t_1 = output.split(self.block_size, 2)
        stack = [t_t.reshape(batch_size, d_height, d_depth) for t_t in t_1]
        output = torch.stack(stack, 1)
        output = output.permute(0, 2, 1, 3)
        output = output.permute(0, 3, 1, 2)
        return output


class DepthToSpace(nn.Module):
    def __init__(self, block_size):
        super(DepthToSpace, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size * block_size

    def forward(self, x):
        output = x.permute(0, 2, 3, 1)
        (batch_size, d_height, d_width, d_depth) = output.size()
        s_depth = int(d_depth / self.block_size_sq)
        s_width = int(d_width * self.block_size)
        s_height = int(d_height * self.block_size)
        t_1 = output.reshape(batch_size, d_height, d_width, self.block_size_sq, s_depth)
        spl = t_1.split(self.block_size, 3)
        stack = [t_t.reshape(batch_size, d_height, s_width, s_depth) for t_t in spl]
        output = torch.stack(stack, 0).transpose(0, 1).permute(0, 2, 1, 3, 4).reshape(batch_size, s_height, s_width,
                                                                                      s_depth)
        output = output.permute(0, 3, 1, 2)
        return output


class Net(nn.Module):
    # layers latent_dim at each layer
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.inplanes = channel = 16

        image_size = config.data.image_size

        init_zero = False
        init_zero_bound = 100

        self.layers = nn.ModuleList()
        cur_layer = 0
        self.n_layers = config.model.n_layers
        subsampling_gap = self.n_layers // (config.model.n_subsampling + 1)
        subsampling_anchors = [subsampling_gap * (i + 1) for i in range(config.model.n_subsampling)]

        self.fc = nn.Linear(self.inplanes * config.data.image_size * config.data.image_size, config.data.num_classes)

        for layer_num in range(self.n_layers):
            if layer_num in subsampling_anchors:
                self.layers.append(SpaceToDepth(2))
                channel *= 2 * 2
                image_size = int(image_size / 2)
                print('space to depth')

            if cur_layer > init_zero_bound:
                init_zero = True

            shape = (channel, image_size, image_size)
            self.layers.append(self._make_layer(shape, 1, config.model.latent_size, channel, init_zero))
            print('basic block')
            

    def _make_layer(self, shape, block_num, latent_dim, input_dim, init_zero):
        layers = []
        for i in range(0, block_num):
            layers.append(BasicBlock(self.config, shape, latent_dim, type='A', input_dim=input_dim,
                                     init_zero=init_zero))
            layers.append(BasicBlock(self.config, shape, latent_dim, type='B', input_dim=input_dim,
                                     init_zero=init_zero))
        return nn.Sequential(*layers)

    def forward(self, x):
        # pad to 16 channels
        x = torch.cat([x, torch.zeros(x.shape[0], 16 - x.shape[1], x.shape[-2], x.shape[-1], device=x.device)], dim=1)

        for layer_num, layer in enumerate(self.layers):
            x = layer(x)

        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

