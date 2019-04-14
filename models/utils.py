def fill_mask(mask, type):
    input_dim = mask.shape[0]
    kernel_mid_y = mask.shape[-2] // 2
    kernel_mid_x = mask.shape[-1] // 2
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


def fill_center_mask(mask):
    input_dim = mask.shape[0]
    kernel_mid_y = mask.shape[-2] // 2
    kernel_mid_x = mask.shape[-1] // 2
    for i in range(input_dim):
        mask[i, i, kernel_mid_y, kernel_mid_x] = 1.0
