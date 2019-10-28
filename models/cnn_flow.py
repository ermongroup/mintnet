import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import math
from .utils import *
import threading
from torch.nn.parallel.parallel_apply import get_a_var, _get_device_index
from itertools import product
from tqdm import tqdm
from numba import jit


def elu_derivative(x, slope=1.0):
    slope1 = torch.ones_like(x)
    x = torch.min(x, torch.ones_like(x) * 70.)
    slope2 = torch.exp(x) * slope
    return torch.where(x > 0, slope1, slope2)


def parallel_apply_sampling(modules, inputs, kwargs_tup=None, devices=None):
    r"""Applies each `module` in :attr:`modules` in parallel on arguments
    contained in :attr:`inputs` (positional) and :attr:`kwargs_tup` (keyword)
    on each of :attr:`devices`.
    Args:
        modules (Module): modules to be parallelized
        inputs (tensor): inputs to the modules
        devices (list of int or torch.device): CUDA devices
    :attr:`modules`, :attr:`inputs`, :attr:`kwargs_tup` (if given), and
    :attr:`devices` (if given) should all have same length. Moreover, each
    element of :attr:`inputs` can either be a single object as the only argument
    to a module, or a collection of positional arguments.
    """
    assert len(modules) == len(inputs)
    if kwargs_tup is not None:
        assert len(modules) == len(kwargs_tup)
    else:
        kwargs_tup = ({},) * len(modules)
    if devices is not None:
        assert len(modules) == len(devices)
    else:
        devices = [None] * len(modules)
    devices = list(map(lambda x: _get_device_index(x, True), devices))
    lock = threading.Lock()
    results = {}
    grad_enabled = torch.is_grad_enabled()

    def _worker(i, module, input, kwargs, device=None):
        torch.set_grad_enabled(grad_enabled)
        if device is None:
            device = get_a_var(input).get_device()
        try:
            with torch.cuda.device(device):
                # this also avoids accidental slicing of `input` if it is a Tensor
                if not isinstance(input, (list, tuple)):
                    input = (input,)
                output = module.sampling(*input, **kwargs)
            with lock:
                results[i] = output
        except Exception as e:
            with lock:
                results[i] = e

    if len(modules) > 1:
        threads = [threading.Thread(target=_worker,
                                    args=(i, module, input, kwargs, device))
                   for i, (module, input, kwargs, device) in
                   enumerate(zip(modules, inputs, kwargs_tup, devices))]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    else:
        _worker(0, modules[0], inputs[0], kwargs_tup[0], devices[0])

    outputs = []
    for i in range(len(inputs)):
        output = results[i]
        if isinstance(output, Exception):
            raise output
        outputs.append(output)
    return outputs


class DataParallelWithSampling(nn.DataParallel):
    def sampling(self, *inputs, **kwargs):
        if not self.device_ids:
            return self.module.sampling(*inputs, **kwargs)
        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        if len(self.device_ids) == 1:
            return self.module.sampling(*inputs[0], **kwargs[0])
        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        outputs = self.parallel_apply_sampling(replicas, inputs, kwargs)
        return self.gather(outputs, self.output_device)

    def parallel_apply_sampling(self, replicas, inputs, kwargs):
        return parallel_apply_sampling(replicas, inputs, kwargs, self.device_ids[:len(replicas)])


class SequentialWithSampling(nn.Sequential):
    def sampling(self, z):
        for module in reversed(self._modules.values()):
            z = module.sampling(z)
        return z


class BasicBlock(nn.Module):
    # Input_dim should be 1(grey scale image) or 3(RGB image), or other dimension if use SpaceToDepth
    def init_conv_weight(self, weight):
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

        self.non_linearity = F.elu
        self.non_linearity_derivative = elu_derivative

        self.t = nn.Parameter(torch.ones(1, *shape))
        self.shape = shape
        self.config = config

    def forward(self, x):
        log_det = x[1]
        x = x[0]
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
        center2 = sign_prods[..., self.kernel3 // 2, self.kernel1 // 2].unsqueeze(-1).unsqueeze(-1) * torch.abs(center2)
        center2 = center2.permute(0, 2, 1, 3, 4, 5).contiguous().view_as(self.weight2)

        masked_weight2 = (center2 * self.center_mask2 + self.weight2 * (1. - self.center_mask2)) * self.mask2

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

        t = torch.max(torch.abs(self.t), torch.tensor(1e-12, device=x.device))
        output = latent_output + t * x
        log_det += torch.sum(torch.log(diag + t), dim=(1, 2, 3))

        return output, log_det

    def sampling(self, z):
        with torch.no_grad():
            masked_weight1 = self.weight1 * self.mask1
            masked_weight3 = self.weight3 * self.mask3
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

            shared_t = torch.max(torch.abs(self.t), torch.tensor(1e-12, device=z.device))

            kernel_mid_y, kernel_mid_x = masked_weight1.shape[-2] // 2, masked_weight1.shape[-1] // 2
            diag1_share = torch.diagonal(
                masked_weight1[..., kernel_mid_y, kernel_mid_x].view(self.latent_dim, self.input_dim,
                                                                     self.input_dim),
                dim1=-2, dim2=-1)[None, :, :, None, None]

            kernel_mid_y, kernel_mid_x = masked_weight2.shape[-2] // 2, masked_weight2.shape[-1] // 2
            diag2_share = masked_weight2[..., kernel_mid_y, kernel_mid_x].view(self.latent_dim, self.input_dim,
                                                                               self.latent_dim,
                                                                               self.input_dim)
            diag2_share = torch.diagonal(diag2_share.permute(0, 2, 1, 3), dim1=-2,
                                         dim2=-1)  # shape: latent_dim x latent_dim x input_dim
            diag2_share = diag2_share[None, :, :, :, None,
                          None]  # shape: 1 x latent_dim x latent_dim x input_dim x 1 x 1

            kernel_mid_y, kernel_mid_x = masked_weight3.shape[-2] // 2, masked_weight3.shape[-1] // 2
            diag3_share = masked_weight3[..., kernel_mid_y, kernel_mid_x].view(self.input_dim, self.latent_dim,
                                                                               self.input_dim)
            diag3_share = torch.diagonal(diag3_share.permute(1, 0, 2), dim1=-2, dim2=-1)[None, :, :, None, None]

            def value_and_grad(x):
                # shape: B x latent_output . input_dim x img_size x img_size
                latent_output = F.conv2d(x, masked_weight1, bias=self.bias1, padding=self.padding1, stride=1)
                diag1 = self.non_linearity_derivative(latent_output). \
                            view(x.shape[0], self.latent_dim, self.input_dim, x.shape[-2], x.shape[-1]) \
                        * diag1_share  # shape: B x latent_dim x input_dim x img_shape x img_shape
                latent_output = self.non_linearity(latent_output)
                latent_output = F.conv2d(latent_output, masked_weight2, bias=self.bias2, padding=self.padding2,
                                         stride=1)
                diag2 = torch.sum(diag2_share * diag1.unsqueeze(1),
                                  dim=2)  # shape: B x latent_dim x input_dim x img_shape x img_shape
                latent_output_derivative = self.non_linearity_derivative(latent_output)
                latent_output = self.non_linearity(latent_output)
                latent_output = F.conv2d(latent_output, masked_weight3, bias=self.bias3, padding=self.padding3,
                                         stride=1)
                diag3 = latent_output_derivative.view(x.shape[0], self.latent_dim, self.input_dim, x.shape[-2],
                                                      x.shape[-1]) \
                        * diag3_share  # shape: B x latent_dim x input_dim x img_shape x img_shape
                diag = torch.sum(diag2 * diag3, dim=1)  # shape: B x input_dim x img_shape x img_shape
                derivative = diag + shared_t  # shape: B x input_dim x img_shape x img_shape
                output = latent_output + shared_t * x  # shape: B x input_dim x img_shape x img_shape
                return output, derivative

            if self.type == 'A':
                print("type A")
                x = z / shared_t  # [0,...]
                for _ in tqdm(range(self.config.model.n_iters)):
                    output, grad = value_and_grad(x)
                    x += (z - output) / (self.config.analysis.newton_lr * grad)
                return x

            elif self.type == 'B':
                print("type B")
                x = z / shared_t  # [0,...]
                for _ in tqdm(range(self.config.model.n_iters)):
                    output, grad = value_and_grad(x)
                    x += (z - output) / (self.config.analysis.newton_lr * grad)
                return x



class SpaceToDepth(nn.Module):
    def __init__(self, block_size):
        super(SpaceToDepth, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size * block_size

    def forward(self, x):
        input = x[0]
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
        return output, x[1]

    def sampling(self, z):
        with torch.no_grad():
            input = z
            output = input.permute(0, 2, 3, 1)
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
        self.inplanes = channel = config.data.channels

        image_size = config.data.image_size
        init_zero = False
        init_zero_bound = config.model.zero_init_start

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
                latent_size //= 2 * 2
                print('space to depth')

            if cur_layer > init_zero_bound:
                init_zero = True

            shape = (channel, image_size, image_size)
            self.layers.append(
                self._make_layer(shape, 1, latent_size, channel, init_zero))
            print('basic block')

        self.sampling_shape = shape

    def _make_layer(self, shape, block_num, latent_dim, input_dim, init_zero):
        layers = []
        for i in range(0, block_num):
            layers.append(BasicBlock(self.config, shape, latent_dim, type='A', input_dim=input_dim,
                                         init_zero=init_zero))

            layers.append(BasicBlock(self.config, shape, latent_dim, type='B', input_dim=input_dim,
                                     init_zero=init_zero))

        return SequentialWithSampling(*layers)

    def forward(self, x):
        log_det = torch.zeros(x.shape[0], device=x.device)

        for layer in self.layers:
            x, log_det = layer([x, log_det])

        x = x.reshape(x.shape[0], -1)
        return x, log_det

    def sampling(self, z):
        z = z.view(z.shape[0], *self.sampling_shape)
        with torch.no_grad():
            for layer in reversed(self.layers):
                z = layer.sampling(z)

            return z