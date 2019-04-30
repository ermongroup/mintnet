import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import math
from .utils import *
from functools import partial


def leaky_relu_derivative(x, slope=0.2):
    slope1 = torch.ones_like(x)
    slope2 = torch.ones_like(x) * slope
    return torch.where(x > 0, slope1, slope2)


def prelu_derivative(x, weight):
    slope1 = torch.ones_like(x)
    slope2 = torch.ones_like(x) * weight
    return torch.where(x > 0, slope1, slope2)


class ActNorm(nn.Module):
    """ An implementation of a activation normalization layer
    from Glow: Generative Flow with Invertible 1x1 Convolutions
    (https://arxiv.org/abs/1807.03039).
    """

    def __init__(self, shape):
        super(ActNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape))
        self.initialized = False

    def forward(self, x):
        inputs = x
        if self.initialized is False:
            self.weight.data.copy_(1 / (inputs.std(0) + 1e-5))
            self.bias.data.copy_(inputs.mean(0))
            self.initialized = True

        return (inputs - self.bias) * self.weight


class FlexibleBasicBlock(nn.Module):
    # Input_dim should be 1(grey scale image) or 3(RGB image), or other dimension if use SpaceToDepth

    def init_conv_weight(self, weight):
        # init.kaiming_uniform_(weight, math.sqrt(5.))
        # init.kaiming_normal_(weight, math.sqrt(5.))
        init.xavier_normal_(weight, 0.005)

    def init_conv_bias(self, weight, bias):
        fan_in, _ = init._calculate_fan_in_and_fan_out(weight)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(bias, -bound, bound)

    def __init__(self, config, shape, latent_dim, type, input_dim=3, kernel1=3, kernel2=3, kernel3=3, init_zero=False):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.padding1 = kernel1 // 2
        self.padding2 = kernel2 // 2
        self.padding3 = kernel3 // 2

        self.weight1 = nn.Parameter(
            torch.randn(input_dim * latent_dim, input_dim, kernel1, kernel1) * 1e-5
        )
        self.bias1 = nn.Parameter(
            torch.zeros(input_dim * latent_dim)
        )

        if not init_zero:
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
        if not init_zero:
            self.init_conv_weight(self.weight3)
            self.init_conv_bias(self.weight3, self.bias3)

        # Define masks

        # Mask out the element above diagonal
        self.type = type
        self.mask1 = nn.Parameter(torch.ones_like(self.weight1), requires_grad=False)
        self.mask2 = nn.Parameter(torch.ones_like(self.weight2), requires_grad=False)
        self.mask3 = nn.Parameter(torch.ones_like(self.weight3), requires_grad=False)

        for i in range(latent_dim):
            fill_mask(self.mask1[i * input_dim: (i + 1) * input_dim, ...], type=type, rgb_last=config.model.rgb_last)
            fill_mask(self.mask3[:, i * input_dim: (i + 1) * input_dim, ...], type=type, rgb_last=config.model.rgb_last)
            for j in range(latent_dim):
                fill_mask(self.mask2[i * input_dim: (i + 1) * input_dim, j * input_dim: (j + 1) * input_dim, ...],
                          type=type, rgb_last=config.model.rgb_last)

        self.non_linearity = partial(F.leaky_relu, negative_slope=0.2)
        self.non_linearity_derivative = partial(leaky_relu_derivative, slope=0.2)

        self.t = nn.Parameter(torch.ones(1, *shape))
        self.shape = shape
        self.config = config

    def forward(self, x):
        log_det = x[1]
        x = x[0]

        ## more flexible diagonal
        masked_weight1 = self.weight1 * self.mask1
        masked_weight3 = self.weight3 * self.mask3

        # shape: B x latent_output . input_dim x img_size x img_size
        latent_output = F.conv2d(x, masked_weight1, bias=self.bias1, padding=self.padding1, stride=1)

        kernel_mid_y, kernel_mid_x = masked_weight1.shape[-2] // 2, masked_weight1.shape[-1] // 2
        diag1 = torch.diagonal(
            masked_weight1[..., kernel_mid_y, kernel_mid_x].view(self.latent_dim, self.input_dim, self.input_dim),
            dim1=-2, dim2=-1)  # shape: latent_dim x input_dim

        diag1 = self.non_linearity_derivative(latent_output). \
                    view(x.shape[0], self.latent_dim, self.input_dim, x.shape[-2], x.shape[-1]) \
                * diag1[None, :, :, None, None]  # shape: B x latent_dim x input_dim x img_shape x img_shape

        latent_output = self.non_linearity(latent_output)

        masked_weight2 = self.weight2 * self.mask2
        ## more flexible diagonal

        latent_output = F.conv2d(latent_output, masked_weight2, bias=self.bias2, padding=self.padding2, stride=1)

        kernel_mid_y, kernel_mid_x = masked_weight2.shape[-2] // 2, masked_weight2.shape[-1] // 2
        diag2 = masked_weight2[..., kernel_mid_y, kernel_mid_x].view(self.latent_dim, self.input_dim, self.latent_dim,
                                                                     self.input_dim)
        diag2 = torch.diagonal(diag2.permute(0, 2, 1, 3), dim1=-2,
                               dim2=-1)  # shape: latent_dim x latent_dim x input_dim
        diag2 = diag2[None, :, :, :, None, None]  # shape: 1 x latent_dim x latent_dim x input_dim x 1 x 1

        diag2 = torch.sum(diag2 * diag1.unsqueeze(1),
                          dim=2)  # shape: B x latent_dim x input_dim x img_shape x img_shape

        latent_output_derivative = self.non_linearity_derivative(latent_output)
        latent_output = self.non_linearity(latent_output)

        latent_output = F.conv2d(latent_output, masked_weight3, bias=self.bias3, padding=self.padding3, stride=1)

        kernel_mid_y, kernel_mid_x = masked_weight3.shape[-2] // 2, masked_weight3.shape[-1] // 2
        diag3 = masked_weight3[..., kernel_mid_y, kernel_mid_x].view(self.input_dim, self.latent_dim, self.input_dim)
        diag3 = torch.diagonal(diag3.permute(1, 0, 2), dim1=-2, dim2=-1)  # shape: latent_dim x input_dim
        diag3 = latent_output_derivative.view(x.shape[0], self.latent_dim, self.input_dim, x.shape[-2], x.shape[-1]) \
                * diag3[None, :, :, None, None]  # shape: B x latent_dim x input_dim x img_shape x img_shape

        diag = torch.sum(diag2 * diag3, dim=1)  # shape: B x input_dim x img_shape x img_shape

        log_det += torch.sum(torch.log(torch.abs(diag + self.t) + 1e-12), dim=(1, 2, 3))

        output = latent_output + self.t * x

        return output, log_det


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


class FlexibleNet(nn.Module):
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
        latent_size = config.model.latent_size

        for layer_num in range(self.n_layers):
            if layer_num in subsampling_anchors:
                self.layers.append(SpaceToDepth(2))
                channel *= 2 * 2
                image_size = int(image_size / 2)
                # Note: do not do latent_size //= 2 * 2 for classification
                print('space to depth')

            if cur_layer > init_zero_bound:
                init_zero = True

            shape = (channel, image_size, image_size)
            self.layers.append(
                self._make_layer(shape, 1, latent_size, channel, init_zero, act_norm=config.model.act_norm,
                                 batch_norm=config.model.batch_norm))
            print('basic block')

        if config.model.act_norm:
            self.pre_fc = nn.Sequential(
                ActNorm(shape),
                nn.ELU()
            )
        else:
            self.pre_fc = nn.ELU()

        self.fc = nn.Linear(shape[0], config.data.num_classes)

    def _make_layer(self, shape, block_num, latent_dim, input_dim, init_zero, act_norm=False, batch_norm=False):
        layers = []
        for i in range(0, block_num):
            if act_norm:
                layers.append(ActNorm(shape))
            if batch_norm:
                layers.append(nn.BatchNorm2d(shape[0]))
            layers.append(FlexibleBasicBlock(self.config, shape, latent_dim, type='A', input_dim=input_dim,
                                             init_zero=init_zero))
            if act_norm:
                layers.append(ActNorm(shape))
            if batch_norm:
                layers.append(nn.BatchNorm2d(shape[0]))
            layers.append(FlexibleBasicBlock(self.config, shape, latent_dim, type='B', input_dim=input_dim,
                                             init_zero=init_zero))
        return nn.Sequential(*layers)

    def forward(self, x):
        # pad to 16 channels
        if not self.config.model.pad_zero:
            x_copy = x.repeat(1, 16 // self.config.data.channels, 1, 1)
            x = torch.cat([x, x_copy[:, :16 - self.config.data.channels, ...]], dim=1)
        else:
            padding = torch.zeros(x.shape[0], 16 - x.shape[1], x.shape[-2], x.shape[-1], device=x.device)
            x = torch.cat([x, padding], dim=1)

        for layer_num, layer in enumerate(self.layers):
            x = layer(x)

        x = self.pre_fc(x)
        # mean pooling
        x = x.mean(dim=(2, 3))
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return F.log_softmax(x, dim=1)
