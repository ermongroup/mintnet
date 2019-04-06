from models.cnn_classification import *
import tensorboardX
import logging
from torchvision.datasets import CIFAR10, MNIST, ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
import numpy as np
import torch.optim as optim


class ClassificationRunner(object):
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


        # Train the model
        step = 0   
        for epoch in range(self.config.training.n_epochs):
            logging.info('Now processing epoch {}'.format(epoch))
            net.train()
            for batch_idx, (data, target) in enumerate(dataloader):            
                data, target = data.to(self.config.device), target.to(self.config.device)
                output = net(data)
                loss = F.nll_loss(output, target)
                pred = output.data.max(1, keepdim=True)[1]
                train_correct = float(pred.eq(target.data.view_as(pred)).sum())/target.shape[0]

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                logging.info("epoch: {}, batch: {}, training_loss: {}, training_accuracy: {}".format(epoch, batch_idx, loss.item(), train_correct))
                tb_logger.add_scalar('training_loss', loss, global_step=step)
                tb_logger.add_scalar('training_accuracy', train_correct, global_step=step)

                step += 1

            # Test
            net.eval()
            with torch.no_grad():
                acc_loss = 0.
                acc_correct = 0
                acc_total = 0
                for batch_idx, (data, target) in enumerate(test_loader):
                    data, target = data.to(self.config.device), target.to(self.config.device)
                    output = net(data)
                    loss = F.nll_loss(output, target)
                    pred = output.data.max(1, keepdim=True)[1]
                    correct = pred.eq(target.data.view_as(pred)).sum()
                    acc_correct += correct
                    acc_total += target.shape[0]
        
                    logging.info("epoch: {}, batch: {}, test_loss: {}, test_accuracy: {}".format(epoch, batch_idx, loss.item(), float(correct)/target.shape[0]))
                    acc_loss += loss.item()
                logging.info("test_loss: {}, test_accuracy: {}".format(acc_loss/(batch_idx + 1), acc_correct/acc_total))
                
                tb_logger.add_scalar('test_loss', acc_loss / (batch_idx + 1), global_step=step)
                tb_logger.add_scalar('test_accuracy', float(acc_correct)/acc_total, global_step=step)

