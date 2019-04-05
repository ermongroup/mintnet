from models.cnn_flow import *
import tensorboardX
import logging
from torchvision.datasets import CIFAR10, MNIST, ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
import numpy as np
import torch.optim as optim


class DensityEstimationRunner(object):
    def __init__(self, args, config):
        self.args = args
        self.config = config

    def get_optimizer(self, parameters):
        if self.config.optim.optimizer == 'Adam':
            return optim.Adam(parameters, lr=self.config.optim.lr, weight_decay=self.config.optim.weight_decay,
                              betas=(self.config.optim.beta1, 0.999))
        elif self.config.optim.optimizer == 'RMSProp':
            return optim.RMSprop(parameters, lr=self.config.optim.lr, weight_decay=self.config.optim.weight_decay)
        elif self.config.optim.optimizer == 'SGD':
            return optim.SGD(parameters, lr=self.config.optim.lr, momentum=0.9)
        else:
            raise NotImplementedError('Optimizer {} not understood.'.format(self.config.optim.optimizer))

    def logit_transform(self, image):
        lambd = self.config.data.lambda_logit
        image = lambd + (1 - 2 * lambd) * image
        return torch.log(image) - torch.log1p(-image)

    def train(self):
        transform = transforms.Compose([
            transforms.Resize(self.config.data.image_size),
            transforms.ToTensor()
        ])

        if self.config.data.dataset == 'CIFAR10':
            dataset = CIFAR10(os.path.join(self.args.run, 'datasets', 'cifar10'), train=True, download=True,
                              transform=transform)
            test_dataset = CIFAR10(os.path.join(self.args.run, 'datasets', 'cifar10'), train=False, download=True,
                                   transform=transform)
        elif self.config.data.dataset == 'MNIST':
            dataset = MNIST(os.path.join(self.args.run, 'datasets', 'mnist'), train=True, download=True,
                            transform=transform)
            num_items = len(dataset)
            indices = list(range(num_items))
            random_state = np.random.get_state()
            np.random.seed(2019)
            np.random.shuffle(indices)
            np.random.set_state(random_state)
            train_indices, test_indices = indices[:int(num_items * 0.8)], indices[int(num_items * 0.8):]
            test_dataset = Subset(dataset, test_indices)
            dataset = Subset(dataset, train_indices)

        elif self.config.data.dataset == 'CELEBA':
            dataset = ImageFolder(root=os.path.join(self.args.run, 'datasets', 'celeba'),
                                  transform=transforms.Compose([
                                      transforms.CenterCrop(140),
                                      transforms.Resize(self.config.data.image_size),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                  ]))
            num_items = len(dataset)
            indices = list(range(num_items))
            random_state = np.random.get_state()
            np.random.seed(2019)
            np.random.shuffle(indices)
            np.random.set_state(random_state)
            train_indices, test_indices = indices[:int(num_items * 0.7)], indices[
                                                                          int(num_items * 0.7):int(num_items * 0.8)]
            test_dataset = Subset(dataset, test_indices)
            dataset = Subset(dataset, train_indices)

        dataloader = DataLoader(dataset, batch_size=self.config.training.batch_size, shuffle=True, num_workers=4,
                                drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=self.config.training.batch_size, shuffle=True,
                                 num_workers=4, drop_last=True)
        test_iter = iter(test_loader)

        net = Net(self.config).to(self.config.device)

        optimizer = self.get_optimizer(net.parameters())
        tb_path = os.path.join(self.args.run, 'tensorboard', self.args.doc)
        tb_logger = tensorboardX.SummaryWriter(log_dir=tb_path)

        def flow_loss(u, log_jacob, size_average=True):
            log_probs = (-0.5 * u.pow(2) - 0.5 * np.log(2 * np.pi)).sum()
            loss = -(log_probs + log_jacob)

            if size_average:
                loss /= u.size(0)
            return loss

        # Train the model
        step = 0
        for epoch in range(self.config.training.n_epochs):
            logging.info('Now processing epoch {}'.format(epoch))

            net.train()
            for batch_idx, (data, target) in enumerate(dataloader):
                # Transform to logit space since pixel values ranging from 0-1
                data = self.logit_transform(data)
                log_det_logit = F.softplus(-data).sum() + F.softplus(data).sum() + np.prod(
                    data.shape) * np.log(1 - 2 * self.config.data.lambda_logit)

                data, target = data.to(self.config.device), target.to(self.config.device)

                output, log_det = net(data)
                loss = flow_loss(output, log_det)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                bpd = (loss.item() * data.shape[0] - log_det_logit) * (1/(np.log(2)*np.prod(data.shape))) + 8
                logging.info("epoch: {}, batch: {}, training_loss: {}, training_bpd: {}".format(epoch, batch_idx, loss.item(), bpd))
                tb_logger.add_scalar('training_loss', loss, global_step=step)
                tb_logger.add_scalar('training_bpd', bpd, global_step=step)

                step += 1

            # Test
            net.eval()
            with torch.no_grad():
                acc_loss = 0.
                acc_bpd = 0.
                for batch_idx, (data, target) in enumerate(test_loader):
                    data = self.logit_transform(data)
                    log_det_logit = F.softplus(-data).sum() + F.softplus(data).sum() + np.prod(
                        data.shape) * np.log(1 - 2 * self.config.data.lambda_logit)

                    data, target = data.to(self.config.device), target.to(self.config.device)
                    output, log_det = net(data)
                    # log_det += log_det_logit
                    loss = flow_loss(output, log_det, False).item()

                    bpd = (loss.item() * data.shape[0] - log_det_logit) * (1/(np.log(2)*np.prod(data.shape))) + 8
                    acc_bpd += bpd
                    logging.info("epoch: {}, batch: {}, test_loss: {}, test_bpd: {}".format(epoch, batch_idx, loss.item(), bpd))
                    acc_loss += loss.item()

                tb_logger.add_scalar('test_loss', acc_loss / (batch_idx + 1), global_step=step)
                tb_logger.add_scalar('test_bpd', acc_bpd / (batch_idx + 1), global_step=step)

