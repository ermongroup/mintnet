from models.cnn_flow import *
from visualization.potential_function import *
from visualization.toy_density import *
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
import shutil
import tensorboardX
import logging
from torchvision.datasets import CIFAR10, MNIST, ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import os
from models.cnn_flow import DataParallelWithSampling
from torchvision.utils import save_image, make_grid
import torch.autograd as autograd
from torch.autograd import Variable
import torch
import matplotlib.pyplot as plt


class SynDensityEstimationRunner(object):
    def __init__(self, args, config):
        self.args = args
        self.config = config

    def get_optimizer(self, parameters):
        if self.config.optim.optimizer == 'Adam':
            return optim.Adam(parameters, lr=self.config.optim.lr, weight_decay=self.config.optim.weight_decay,
                              betas=(self.config.optim.beta1, 0.999), amsgrad=self.config.optim.amsgrad,
                              eps=self.config.optim.adam_eps)
        elif self.config.optim.optimizer == 'RMSProp':
            return optim.RMSprop(parameters, lr=self.config.optim.lr, weight_decay=self.config.optim.weight_decay)
        elif self.config.optim.optimizer == 'SGD':
            return optim.SGD(parameters, lr=self.config.optim.lr, momentum=0.9)
        elif self.config.optim.optimizer == 'Adamax':
            return optim.Adamax(parameters, lr=self.config.optim.lr, betas=(self.config.optim.beta1, 0.999),
                                weight_decay=self.config.optim.weight_decay)
        else:
            raise NotImplementedError('Optimizer {} not understood.'.format(self.config.optim.optimizer))


    def generate_data(self, partition=10, train=True):
        if train:
            size = self.config.training.batch_size
        else:
            size = self.config.training.evaluation

        if self.config.data.toy == 1:
            assert(size % partition == 0)
            data = toy1(partition=partition, radius=15, size=int(size/partition))
        elif self.config.data.toy == 2:
            data = toy2(num=5, radius=5, interval=5, size=size)
        elif self.config.data.toy == 3:
            data = toy3(size=size)

        data = torch.Tensor(data)
        data = data.reshape((size, self.config.data.channels, 1, 1))
        return data

    def train(self):
        net = Net(self.config).to(self.config.device)
        net = DataParallelWithSampling(net)
        optimizer = self.get_optimizer(net.parameters())

        def smoothplot(x, y, s, bins=1000):
            from scipy.ndimage.filters import gaussian_filter
            heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins)
            heatmap = gaussian_filter(heatmap, sigma=s)
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
            return heatmap.T, extent


        def flow_loss(u, log_jacob, size_average=True):
            log_probs = (-0.5 * u.pow(2) - 0.5 * np.log(2 * np.pi)).sum()
            log_jacob = log_jacob.sum()
            loss = -(log_probs + log_jacob)
            if size_average:
                loss /= u.size(0)
            return loss

        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.config.training.n_epochs, eta_min=0.)

        # Train the model
        step = 0
        for epoch in range(0, self.config.training.n_epochs):
            scheduler.step()
            data = self.generate_data()
            '''
            import pdb
            samples = data.to(device='cpu').numpy()
            samples = samples.reshape((samples.shape[0], samples.shape[1]))
            # sigmas = [0, 16, 32, 64]
            x = samples[:, 0]
            y = samples[:, 1]
            img, extent = smoothplot(x, y, 1)
            plt.imshow(img, extent=extent, origin='lower')
            plt.savefig('samples-syn-forward-check.png')
            pdb.set_trace()
            '''

            net.train()
            output, log_det = net(data)
            loss = flow_loss(output, log_det)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % self.config.training.snapshot_interval == 0:
                states = [
                    net.state_dict(),
                    optimizer.state_dict(),
                    epoch + 1,
                    step,
                    scheduler.state_dict()
                ]
                torch.save(states, os.path.join(self.args.run, 'logs', self.args.doc,
                                                'checkpoint_epoch_{}.pth'.format(epoch + 1)))
                torch.save(states, os.path.join(self.args.run, 'logs', self.args.doc, 'checkpoint.pth'))

            if step % self.config.training.log_interval == 0:
                logging.info("epoch: {}, training_loss: {}".format(epoch, loss.item()))
            step += 1


    def Langevin_dynamics(self, x_mod, net, n_steps=200, step_lr=0.00005):
        images = []
        def log_prob(x):
            u, log_jacob = net(x)
            log_probs = (-0.5 * u.pow(2) - 0.5 * np.log(2 * np.pi)).sum()
            log_jacob = log_jacob.sum()
            loss = (log_probs + log_jacob)
            return loss

        def score(x):
            with torch.enable_grad():
                x.requires_grad_(True)
                return autograd.grad(log_prob(x), x)[0]

        with torch.no_grad():
            for _ in range(n_steps):
                images.append(torch.clamp(x_mod, 0.0, 1.0))
                noise = torch.randn_like(x_mod) * np.sqrt(step_lr * 2)
                grad = score(x_mod)
                x_mod = x_mod + step_lr * grad + noise
                x_mod = x_mod
                print("modulus of grad components: mean {}, max {}".format(grad.abs().mean(), grad.abs().max()))
            return images


    def test(self):
        test_data = self.generate_data(train=False)
        net = Net(self.config).to(self.config.device)
        net = DataParallelWithSampling(net)
        optimizer = self.get_optimizer(net.parameters())

        def smoothplot(x, y, s, bins=1000):
            from scipy.ndimage.filters import gaussian_filter
            heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins)
            heatmap = gaussian_filter(heatmap, sigma=s)
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
            return heatmap.T, extent

        def flow_loss(u, log_jacob, size_average=True):
            log_probs = (-0.5 * u.pow(2) - 0.5 * np.log(2 * np.pi)).sum()
            log_jacob = log_jacob.sum()
            loss = -(log_probs + log_jacob)

            if size_average:
                loss /= u.size(0)
            return loss

        states = torch.load(os.path.join(self.args.run, 'logs', self.args.doc, 'checkpoint.pth'),
                            map_location=self.config.device)

        net.load_state_dict(states[0])
        optimizer.load_state_dict(states[1])
        loaded_epoch = states[2]

        logging.info(
            "Loading the model from epoch {}".format(loaded_epoch))

        # Test the model
        net.eval()
        total_loss = 0
        total_n_data = 0
        logging.info("Generating samples")

        #samples = test_data.to(device='cpu').numpy()
        #samples = samples.reshape((samples.shape[0], samples.shape[1]))
        x = []
        y = []

        for i in range(5):
            z = torch.randn(10000, self.config.data.channels * self.config.data.image_size * self.config.data.image_size,
                        device=self.config.device)
            samples = net.sampling(z)
            samples = samples.to(device='cpu').numpy()
            samples = samples.reshape((samples.shape[0], samples.shape[1]))
            x.extend(samples[:, 0])
            y.extend(samples[:, 1])

        img, extent = smoothplot(x, y, 1)
        plt.imshow(img, extent=extent, origin='lower')
        plt.savefig('samples-syn-forward.png')

        with torch.no_grad():
            test_output, test_log_det = net(test_data)
            test_loss = flow_loss(test_output, test_log_det)
            total_loss += test_loss * test_data.shape[0]
            total_n_data += test_data.shape[0]

        logging.info(
        "Total loss: {}\n".format(total_loss.item() / total_n_data))
