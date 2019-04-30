import torch
import math

def potential_1(z):
    z1, z2 = torch.chunk(z, chunks=2, dim=1)
    norm = torch.sqrt(z1 ** 2 + z2 ** 2)
    exp1 = torch.exp(-0.5 * ((z1 - 2) / 0.6) ** 2)
    exp2 = torch.exp(-0.5 * ((z1 + 2) / 0.6) ** 2)
    u = 0.5 * ((norm - 2) / 0.4) ** 2 - torch.log(exp1 + exp2)
    return torch.exp(-u)


def potential_2(z):
    z1, z2 = torch.chunk(z, chunks=2, dim=1)
    w1 = torch.sin((2 * math.pi * z1) / 4)
    u = 0.5 * ((z2 - w1) / 0.4) ** 2
    return torch.exp(-u)


def potential_3(z):
    z1, z2 = torch.chunk(z, chunks=2, dim=1)
    w1 = torch.sin((2 * math.pi * z1) / 4)
    w2 = 3. * torch.exp(-0.5 * ((z1 - 1) / 0.6) ** 2)
    exp1 = torch.exp(-0.5 * ((z2 - w1) / 0.35) ** 2)
    exp2 = torch.exp(-0.5 * ((z2 - w1 + w2) / 0.35) ** 2)
    u = exp1 + exp2
    return u


def potential_4(z):
    z1, z2 = torch.chunk(z, chunks=2, dim=1)
    w1 = torch.sin((2 * math.pi * z1) / 4)
    w3 = 3. * torch.sigmoid((z1 - 1) / 0.3)
    exp1 = torch.exp(-0.5 * ((z2 - w1) / 0.4) ** 2)
    exp2 = torch.exp(-0.5 * ((z2 - w1 + w3) / 0.35) ** 2)
    u = exp1 + exp2
    return u