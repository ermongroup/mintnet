from numba import jit
import torch
import torch.nn as nn
import copy

@jit(nopython=True)
def fill_mask(mask, type, rgb_last=True):
    input_dim = mask.shape[0]
    kernel_mid_y = mask.shape[-2] // 2
    kernel_mid_x = mask.shape[-1] // 2
    if rgb_last:
        if type == 'A':
            for i in range(input_dim):
                mask[i, i + 1:, :, :] = 0.0
                mask[i, i, kernel_mid_y + 1:, :] = 0.0
                mask[i, i, kernel_mid_y, kernel_mid_x + 1:] = 0.0
        if type == 'B':
            for i in range(input_dim):
                mask[i, :i, :, :] = 0.0
                mask[i, i, :kernel_mid_y, :] = 0.0
                mask[i, i, kernel_mid_y, :kernel_mid_x] = 0.0
    else:
        if type == 'A':
            for i in range(input_dim):
                mask[i, :, kernel_mid_y + 1:, :] = 0.0
                # For the current and previous color channels, including the current color
                mask[i, :i + 1, kernel_mid_y, kernel_mid_x + 1:] = 0.0
                # For the latter color channels, not including the current color
                mask[i, i + 1:, kernel_mid_y, kernel_mid_x:] = 0.0

        elif type == 'B':
            for i in range(input_dim):
                mask[i, :, :kernel_mid_y, :] = 0.0
                # For the current and latter color channels, including the current color
                mask[i, i:, kernel_mid_y, :kernel_mid_x] = 0.0
                # For the previous color channels, not including the current color
                mask[i, :i, kernel_mid_y, :kernel_mid_x + 1] = 0.0
        else:
            raise TypeError('type should be either A or B')

@jit(nopython=True)
def fill_center_mask(mask):
    input_dim = mask.shape[0]
    kernel_mid_y = mask.shape[-2] // 2
    kernel_mid_x = mask.shape[-1] // 2
    for i in range(input_dim):
        mask[i, i, kernel_mid_y, kernel_mid_x] = 1.0

@jit(nopython=True)
def generate_masks(mask1, center_mask1, mask2, center_mask2, mask3, center_mask3, input_dim, latent_dim, type, rgb_last):
    for i in range(latent_dim):
        fill_mask(mask1[i * input_dim: (i + 1) * input_dim, ...], type=type, rgb_last=rgb_last)
        fill_center_mask(center_mask1[i * input_dim: (i + 1) * input_dim, ...])
        fill_mask(mask3[:, i * input_dim: (i + 1) * input_dim, ...], type=type, rgb_last=rgb_last)
        fill_center_mask(center_mask3[:, i * input_dim: (i + 1) * input_dim, ...])
        for j in range(latent_dim):
            fill_mask(mask2[i * input_dim: (i + 1) * input_dim, j * input_dim: (j + 1) * input_dim, ...],
                      type=type, rgb_last=rgb_last)
            fill_center_mask(
                center_mask2[i * input_dim: (i + 1) * input_dim, j * input_dim: (j + 1) * input_dim, ...])


class EMAHelper(object):
    def __init__(self, mu=0.999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = torch.zeros_like(param, requires_grad=False)

    def update(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        module_copy = copy.deepcopy(module)
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict
