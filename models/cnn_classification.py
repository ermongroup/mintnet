import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import math
from .utils import *
import tqdm
from functools import partial


class BasicBlock(nn.Module):
    # Input_dim should be 1(grey scale image) or 3(RGB image), or other dimension if use SpaceToDepth
    def init_conv_weight(self, weight):
        init.kaiming_uniform_(weight, math.sqrt(5.))

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
        self.kernel1 = kernel1
        self.kernel2 = kernel2
        self.kernel3 = kernel3

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
        )
        self.bias2 = nn.Parameter(
            torch.zeros(input_dim * latent_dim)
        )

        if not init_zero:
            self.init_conv_weight(self.weight2)
            self.init_conv_bias(self.weight2, self.bias2)

        self.weight3 = nn.Parameter(
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
        self.mask1 = np.ones(self.weight1.shape, dtype=np.float32)
        self.center_mask1 = np.zeros(self.weight1.shape, dtype=np.float32)
        self.mask2 = np.ones(self.weight2.shape, dtype=np.float32)
        self.center_mask2 = np.zeros(self.weight2.shape, dtype=np.float32)
        self.mask3 = np.ones(self.weight3.shape, dtype=np.float32)
        self.center_mask3 = np.zeros(self.weight3.shape, dtype=np.float32)

        generate_masks(self.mask1, self.center_mask1, self.mask2, self.center_mask2, self.mask3, self.center_mask3,
                       input_dim, latent_dim, type, config.model.rgb_last)

        self.mask1 = nn.Parameter(torch.from_numpy(self.mask1), requires_grad=False)
        self.center_mask1 = nn.Parameter(torch.from_numpy(self.center_mask1), requires_grad=False)
        self.mask2 = nn.Parameter(torch.from_numpy(self.mask2), requires_grad=False)
        self.center_mask2 = nn.Parameter(torch.from_numpy(self.center_mask2), requires_grad=False)
        self.mask3 = nn.Parameter(torch.from_numpy(self.mask3), requires_grad=False)
        self.center_mask3 = nn.Parameter(torch.from_numpy(self.center_mask3), requires_grad=False)

        self.non_linearity = F.leaky_relu

        self.t = nn.Parameter(torch.ones(1, *shape))
        self.shape = shape
        self.config = config


   
    def forward(self, x):
        masked_weight1 = self.weight1 * self.mask1
        masked_weight3 = self.weight3 * self.mask3

        # shape: B x latent_output . input_dim x img_size x img_size
        latent_output = F.conv2d(x, masked_weight1, bias=self.bias1, padding=self.padding1, stride=1)
        latent_output = self.non_linearity(latent_output)

        center1 = masked_weight1 * self.center_mask1  # shape: latent_dim.input_dim x input_dim x kernel x kernel
        center3 = masked_weight3 * self.center_mask3  # shape: input_dim x latent_dim.input_dim x kernel x kernel

        # shape: 1 x latent_dim x input_dim x input_dim x kernel x kernel
        center1 = center1.view(self.latent_dim, self.input_dim, self.input_dim,
                               center1.shape[-2], center1.shape[-1]).unsqueeze(0)
        # shape: latent_dim x 1 x input_dim x input_dim x kernel x kernel
        center3 = center3.view(self.input_dim, self.latent_dim, self.input_dim, center3.shape[-2],
                               center3.shape[-1]).permute(1, 0, 2, 3, 4).unsqueeze(1)

        sign_prods = torch.sign(center1) * torch.sign(center3)
        center2 = self.weight2 * self.center_mask2  # shape: latent_dim.input_dim x latent_dim.input_dim x kernel x kernel
        center2 = center2.view(self.latent_dim, self.input_dim, self.latent_dim, self.input_dim,
                               center2.shape[-2], center2.shape[-1])

        center2 = center2.permute(0, 2, 1, 3, 4, 5)
        center2 = sign_prods * torch.abs(center2)
        center2 = center2.permute(0, 2, 1, 3, 4, 5).contiguous().view_as(self.weight2)
        masked_weight2 = (center2 * self.center_mask2 + self.weight2 * (1. - self.center_mask2)) * self.mask2

        latent_output = F.conv2d(latent_output, masked_weight2, bias=self.bias2, padding=self.padding2, stride=1)
        latent_output = self.non_linearity(latent_output)

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
                self._make_layer(shape, 1, latent_size, channel, init_zero,
                                 batch_norm=config.model.batch_norm))
            print('basic block')

        self.pre_fc = nn.ELU()
        self.fc = nn.Linear(shape[0], config.data.num_classes)

    def _make_layer(self, shape, block_num, latent_dim, input_dim, init_zero, batch_norm=False):
        layers = []
        for i in range(0, block_num):
            if batch_norm:
                layers.append(nn.BatchNorm2d(shape[0]))
            layers.append(BasicBlock(self.config, shape, latent_dim, type='A', input_dim=input_dim,
                                     init_zero=init_zero))
            if batch_norm:
                layers.append(nn.BatchNorm2d(shape[0]))
            layers.append(BasicBlock(self.config, shape, latent_dim, type='B', input_dim=input_dim,
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
