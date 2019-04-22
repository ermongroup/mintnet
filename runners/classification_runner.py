from models.cnn_classification import *
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


class ClassificationRunner(object):
    def __init__(self, args, config):
        self.args = args
        self.config = config

    def get_optimizer(self, parameters):
        if self.config.optim.optimizer == 'Adam':
            return optim.Adam(parameters, lr=self.config.optim.lr, weight_decay=self.config.optim.weight_decay,
                              betas=(self.config.optim.beta1, 0.999), amsgrad=self.config.optim.amsgrad)
        elif self.config.optim.optimizer == 'RMSProp':
            return optim.RMSprop(parameters, lr=self.config.optim.lr, weight_decay=self.config.optim.weight_decay)
        elif self.config.optim.optimizer == 'SGD':
            return optim.SGD(parameters, lr=self.config.optim.lr, momentum=0.9)
        else:
            raise NotImplementedError('Optimizer {} not understood.'.format(self.config.optim.optimizer))

    def train(self):

        if self.config.data.dataset == 'CIFAR10':
            transform = transforms.Compose([
                transforms.Resize(self.config.data.image_size),
                transforms.ToTensor(),
                transforms.Normalize(self.config.data.mean, self.config.data.std)
            ])

            dataset = CIFAR10(os.path.join(self.args.run, 'datasets', 'cifar10'), train=True, download=True,
                              transform=transform)
            test_dataset = CIFAR10(os.path.join(self.args.run, 'datasets', 'cifar10'), train=False, download=True,
                                   transform=transform)

        elif self.config.data.dataset == 'MNIST':
            transform = transforms.Compose([
                transforms.Resize(self.config.data.image_size),
                transforms.ToTensor()
            ])

            dataset = MNIST(os.path.join(self.args.run, 'datasets', 'mnist'), train=True, download=True,
                            transform=transform)
            test_dataset = MNIST(os.path.join(self.args.run, 'datasets', 'mnist_test'), train=False, download=True,
                                 transform=transform)

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
        test_loader = DataLoader(test_dataset, batch_size=self.config.training.batch_size, shuffle=False,
                                 num_workers=4, drop_last=True)
        test_iter = iter(test_loader)

        net = Net(self.config).to(self.config.device)
        net = torch.nn.DataParallel(net)
        optimizer = self.get_optimizer(net.parameters())

        tb_path = os.path.join(self.args.run, 'tensorboard', self.args.doc)
        if os.path.exists(tb_path):
            shutil.rmtree(tb_path)

        tb_logger = tensorboardX.SummaryWriter(log_dir=tb_path)

        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.run, 'logs', self.args.doc, 'checkpoint_epoch_90.pth'),
                                map_location=self.config.device)
            net.load_state_dict(states[0])
            optimizer.load_state_dict(states[1])
            begin_epoch = states[2]
            step = states[3]
        else:
            step = 0
            begin_epoch = 0

        # Train the model
        # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150], gamma=0.3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.config.training.n_epochs,
                                                               eta_min=1e-6, last_epoch=-1)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=1e-08)

        for epoch in range(begin_epoch, self.config.training.n_epochs):
            scheduler.step()
            # manually adjust learning rate
            # self.adjust_learning_rate(optimizer, epoch)
            # total_loss = 0 #for plateau scheduler only
            for batch_idx, (data, target) in enumerate(dataloader):
                net.train()
                data = data.to(device=self.config.device)
                target = target.to(device=self.config.device)
                output = net(data)
                loss = F.nll_loss(output, target)

                pred = torch.argmax(output, dim=1, keepdim=True)
                train_accuracy = float(pred.eq(target.data.view_as(pred)).sum()) / float(target.shape[0])

                # total_loss += loss.data #for plateau scheduler
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # validation
                net.eval()

                with torch.no_grad():
                    try:
                        test_data, test_target = next(test_iter)
                    except:
                        test_iter = iter(test_loader)
                        test_data, test_target = next(test_iter)

                    test_data = test_data.to(device=self.config.device)
                    test_target = test_target.to(device=self.config.device)
                    test_output = net(test_data)
                    test_loss = F.nll_loss(test_output, test_target)
                    test_pred = torch.argmax(test_output, dim=1, keepdim=True)
                    test_accuracy = float(test_pred.eq(test_target.data.view_as(test_pred)).sum()) / test_data.shape[0]

                tb_logger.add_scalar('training_loss', loss, global_step=step)
                tb_logger.add_scalar('training_accuracy', train_accuracy, global_step=step)
                tb_logger.add_scalar('test_loss', test_loss, global_step=step)
                tb_logger.add_scalar('test_accuracy', test_accuracy, global_step=step)

                if step % self.config.training.log_interval == 0:
                    logging.info(
                        "epoch: {}, batch: {}, training_loss: {}, train_accuracy: {}, test_loss: {}, test_accuracy: {}"
                            .format(epoch, batch_idx, loss.item(), train_accuracy, test_loss.item(), test_accuracy))
                step += 1

            # scheduler.step(total_loss) #for palteau scheduler only
            if (epoch + 1) % self.config.training.snapshot_interval == 0:
                print(self.config.training.snapshot_interval)
                states = [
                    net.state_dict(),
                    optimizer.state_dict(),
                    epoch + 1,
                    step
                ]
                torch.save(states, os.path.join(self.args.run, 'logs', self.args.doc,
                                                'checkpoint_epoch_{}.pth'.format(epoch + 1)))
                torch.save(states, os.path.join(self.args.run, 'logs', self.args.doc, 'checkpoint.pth'))
