# instead of sharing the middle weights for the kernal, we take the softplus/abs of the middle weights to make sure
# the product is none negative

import os
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import math


def leaky_relu_derivative(x, slope):
    slope1 = torch.ones_like(x).detach()
    slope2 = torch.ones_like(x).detach() * slope
    return torch.where(x > 0, slope1, slope2).detach()


def elu_derivative(x, slope):
    slope1 = torch.ones_like(x).detach()
    slope2 = torch.exp(x).detach() * slope
    return torch.where(x > 0, slope1, slope2).detach()


def leaky_relu(x, log_det, slope):
    slope1 = torch.ones_like(x).detach()
    slope2 = torch.ones_like(x).detach() * slope
    x = F.leaky_relu(x, negative_slope=slope)
    log_det += torch.sum(torch.log(torch.where(x > 0, slope1, slope2)))
    return x, log_det


# invertible batch norm 
# see paper: Masked Autoregressive Flow for Density Estimation
EPSILON = 1e-8
'''
def batch_norm(output, log_det):
    m = output.mean().item()
    v = output.var().item()
    u = (output - m) /np.sqrt(v + EPSILON)
    log_det += np.sum(- 0.5 * np.log(v + EPSILON))
    return u, log_det
'''


def batch_norm(output, log_det, beta, gamma):
    m = output.mean().item()
    v = output.var().item()
    u = (output - m) * torch.exp(gamma) / np.sqrt(v + EPSILON) + beta
    log_det += torch.sum(gamma - 0.5 * np.log(v + EPSILON))
    return u, log_det


# DO NOT FORGET ACTNORM!!!
class BasicBlock(nn.Module):
    # Input_dim should be 1(grey scale image) or 3(RGB image), or other dimension if use SpaceToDepth

    def init_conv_weight(self, weight):
        init.kaiming_uniform_(weight, a=math.sqrt(5))

    def init_conv_bias(self, weight, bias):
        fan_in, _ = init._calculate_fan_in_and_fan_out(weight)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(bias, -bound, bound)

    def __init__(self, config, latent_dim, type, input_dim=3, kernel=3, padding=1, stride=1):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.res = nn.Parameter(torch.ones(1))

        self.weight1 = nn.Parameter(
            torch.zeros(input_dim * latent_dim, input_dim, kernel, kernel, device=config.device)
        )
        self.bias1 = nn.Parameter(
            torch.zeros(input_dim * latent_dim, device=config.device)
        )
        self.center1 = nn.Parameter(
            torch.randn(input_dim * latent_dim, input_dim, kernel, kernel, device=config.device)
        )
        self.init_conv_weight(self.weight1)
        self.init_conv_bias(self.weight1, self.bias1)
        self.init_conv_weight(self.center1)

        self.weight2 = nn.Parameter(
            torch.zeros(input_dim * latent_dim, input_dim, kernel, kernel, device=config.device)
        )
        self.bias2 = nn.Parameter(
            torch.zeros(input_dim * latent_dim, device=config.device)
        )
        self.center2 = nn.Parameter(
            torch.randn(input_dim * latent_dim, input_dim, kernel, kernel, device=config.device)
        )

        self.init_conv_weight(self.weight2)
        self.init_conv_bias(self.weight2, self.bias2)
        self.init_conv_weight(self.center2)

        # Define masks
        kernel_mid_y, kernel_mid_x = kernel // 2, kernel // 2
        # zero in the middle(technically not middle, depending on channels), one elsewhere
        # used to mask out the diagonal element
        self.mask0 = np.ones((input_dim, input_dim, kernel, kernel), dtype=np.float32)

        # 1 in the middle, zero elsewhere, used for center mask to zero out the non-diagonal element
        self.mask1 = np.zeros((input_dim, input_dim, kernel, kernel), dtype=np.float32)

        # Mask out the element above diagonal
        self.mask = np.ones((input_dim, input_dim, kernel, kernel), dtype=np.float32)

        # For RGB ONLY:i=0:Red channel;i=1:Green channel;i=2:Blue channel
        if type == 'A':
            for i in range(input_dim):
                self.mask0[i, i, kernel_mid_y, kernel_mid_x] = 0.0
                self.mask1[i, i, kernel_mid_y, kernel_mid_x] = 1.0
                self.mask[i, :, kernel_mid_y + 1:, :] = 0.0
                # For the current and previous color channels, including the current color
                self.mask[i, :i + 1, kernel_mid_y, kernel_mid_x + 1:] = 0.0
                # For the latter color channels, not including the current color
                self.mask[i, i + 1:, kernel_mid_y, kernel_mid_x:] = 0.0
        elif type == 'B':
            for i in range(input_dim):
                self.mask0[i, i, kernel_mid_y, kernel_mid_x] = 0.0
                self.mask1[i, i, kernel_mid_y, kernel_mid_x] = 1.0
                self.mask[i, :, :kernel_mid_y, :] = 0.0
                # For the current and latter color channels, including the current color
                self.mask[i, i:, kernel_mid_y, :kernel_mid_x] = 0.0
                # For the previous color channels, not including the current color
                self.mask[i, :i, kernel_mid_y, :kernel_mid_x + 1] = 0.0
        else:
            raise TypeError('type should be either A or B')

        self.mask0 = torch.tensor(self.mask0, device=config.device)
        self.mask1 = torch.tensor(self.mask1, device=config.device)
        self.mask = torch.tensor(self.mask, device=config.device)

        self.mask0 = self.mask0.repeat(latent_dim, 1, 1, 1)
        self.mask1 = self.mask1.repeat(latent_dim, 1, 1, 1)
        self.mask = self.mask.repeat(latent_dim, 1, 1, 1)

    def forward(self, x):
        log_det = x[1]
        x = x[0]
        residual = x

        masked_weight1 = (self.weight1 * self.mask0 + F.softplus(self.center1) * self.mask1) * self.mask
        latent_output = F.conv2d(x.repeat(1, self.latent_dim, 1, 1), masked_weight1, bias=self.bias1,
                                 padding=self.padding, stride=self.stride,
                                 groups=self.latent_dim)

        center1_diag = self.center1.view(self.latent_dim, self.input_dim, self.input_dim, self.kernel, self.kernel)
        center1_diag = torch.diagonal(center1_diag[..., self.kernel // 2, self.kernel // 2], dim1=-2, dim2=-1)
        center1_diag = F.softplus(center1_diag)

        center2_diag = self.center2.view(self.latent_dim, self.input_dim, self.input_dim, self.kernel, self.kernel)
        center2_diag = torch.diagonal(center2_diag[..., self.kernel // 2, self.kernel // 2], dim1=-2, dim2=-1)
        center2_diag = F.softplus(center2_diag)

        center_diag = center1_diag * center2_diag  # shape: latent_dim x input_dim
        latent_output_elu_derivative = elu_derivative(latent_output,
                                                      1)  # shape: B x latent_dim . input_dim x kernel x kernel
        latent_output_elu_derivative = latent_output_elu_derivative.view(-1, self.latent_dim, self.input_dim,
                                                                         latent_output.shape[-2],
                                                                         latent_output.shape[-1])

        self.diag = (center_diag[..., None, None] * latent_output_elu_derivative).sum(1)
        latent1 = F.elu(latent_output, alpha=1)

        masked_weight2 = (self.weight2 * self.mask0 + F.softplus(self.center2) * self.mask1) * self.mask
        latent2 = F.conv2d(latent1, masked_weight2, bias=self.bias2, padding=self.padding, stride=self.stride,
                           groups=self.latent_dim)

        output = latent2.sum(dim=1, keepdim=True) / latent2.shape[1]
        mask_res = (self.res > 0).float().to(x.device)

        # MIGHT NEED TO ADD EPSILON TO self.res * mask_res
        output = output + self.res * mask_res * residual
        self.diag = self.diag / latent2.shape[1] + self.res * mask_res
        log_det += torch.sum(torch.log(self.diag))

        # need to act_norm

        return output, log_det


class SpaceToDepth(nn.Module):
    def __init__(self, block_size):
        super(SpaceToDepth, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size * block_size

    def forward(self, input):
        output = input.permute(0, 2, 3, 1)
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


class Net(nn.Module):
    # layers latent_dim at each layer
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.inplanes = channel = config.data.channels
        self.increase_dim = SpaceToDepth(4)
        layer_size = config.model.layer_size
        latent_size = config.model.latent_size
        # self.increase_dim = SpaceToDepth(2)
        self.layer1 = self._make_layer(layer_size[0], latent_size[0], channel)
        # channel *= 2 * 2
        # channel *= 4 * 4
        self.layer2 = self._make_layer(layer_size[1], latent_size[1], channel)
        self.layer3 = self._make_layer(layer_size[2], latent_size[2], channel)

    def _make_layer(self, block_num, latent_dim, input_dim, stride=1):
        layers = []
        for i in range(0, block_num):
            layers.append(BasicBlock(self.config, latent_dim, type='A', input_dim=input_dim))
            layers.append(BasicBlock(self.config, latent_dim, type='B', input_dim=input_dim))
        return nn.Sequential(*layers)

    def forward(self, x):
        log_det = torch.zeros([1], device=x.device)
        x, log_det = self.layer1([x, log_det])
        # x, log_det = leaky_relu(x, log_det, 0.01)
        # x = self.increase_dim(x)
        x, log_det = self.layer2([x, log_det])
        # x, log_det = leaky_relu(x, log_det, 0.01)
        x, log_det = self.layer3([x, log_det])
        # x, log_det = leaky_relu(x, log_det, 0.01)
        # x, log_det = self.layer4([x, log_det])
        x = x.view(x.shape[0], -1)
        return x, log_det
