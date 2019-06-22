from models.cnn_flow import Net
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
from datasets.imagenet import OordImageNet
import torch.autograd as autograd
import torch
import matplotlib.pyplot as plt
import tqdm
import seaborn as sns
import  math
import pickle
from models.utils import EMAHelper
sns.set()


class DensityEstimationRunner(object):
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

    def logit_transform(self, image):
        lambd = self.config.data.lambda_logit
        image = lambd + (1 - 2 * lambd) * image
        return torch.log(image) - torch.log1p(-image)

    def sigmoid_transform(self, samples):
        lambd = self.config.data.lambda_logit
        samples = torch.sigmoid(samples)
        samples = (samples - lambd) / (1 - 2 * lambd)
        return samples

    def compute_grad_norm(self, model):
        # total_norm = 0.
        # for p in model.parameters():
        #     if p.requires_grad is True:
        #         total_norm += p.grad.data.norm().item() ** 2
        # return total_norm ** (1 / 2.)
        minv = np.inf
        maxv = -np.inf
        meanv = 0.
        total_p = 0
        for p in model.parameters():
            if p.requires_grad is True:
                minv = min(minv, p.grad.data.abs().min().item())
                maxv = max(maxv, p.grad.data.abs().max().item())
                meanv += p.grad.data.abs().sum().item()
                total_p += np.prod(p.grad.data.shape)
        return minv, maxv, meanv / total_p

    def train(self):
        if self.config.data.horizontal_flip:
            train_transform = transforms.Compose([
                transforms.Resize(self.config.data.image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor()
            ])

        else:
            train_transform = transforms.Compose([
                transforms.Resize(self.config.data.image_size),
                transforms.ToTensor()
            ])

        test_transform = transforms.Compose([
            transforms.Resize(self.config.data.image_size),
            transforms.ToTensor()
        ])

        if self.config.data.dataset == 'CIFAR10':
            dataset = CIFAR10(os.path.join(self.args.run, 'datasets', 'cifar10'), train=True, download=True,
                              transform=train_transform)
            test_dataset = CIFAR10(os.path.join(self.args.run, 'datasets', 'cifar10'), train=False, download=True,
                                   transform=test_transform)
        elif self.config.data.dataset == 'MNIST':
            dataset = MNIST(os.path.join(self.args.run, 'datasets', 'mnist'), train=True, download=True,
                            transform=train_transform)
            test_dataset = MNIST(os.path.join(self.args.run, 'datasets', 'mnist_test'), train=False, download=True,
                                 transform=test_transform)
        elif self.config.data.dataset == 'ImageNet':
            dataset = OordImageNet('/atlas/u/yangsong/datasets/oord_imagenet', train=True, transform=train_transform)
            test_dataset = OordImageNet('/atlas/u/yangsong/datasets/oord_imagenet', train=False, transform=test_transform)

        dataloader = DataLoader(dataset, batch_size=self.config.training.batch_size, shuffle=True, num_workers=4,
                                drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=self.config.training.batch_size, shuffle=True,
                                 num_workers=4, drop_last=True)
        test_iter = iter(test_loader)

        net = Net(self.config).to(self.config.device)
        net = DataParallelWithSampling(net)
        ema_helper = EMAHelper(mu=0.999)
        ema_helper.register(net)

        optimizer = self.get_optimizer(net.parameters())

        tb_path = os.path.join(self.args.run, 'tensorboard', self.args.doc)
        if os.path.exists(tb_path):
            shutil.rmtree(tb_path)

        tb_logger = tensorboardX.SummaryWriter(logdir=tb_path)

        def flow_loss(u, log_jacob, size_average=True):
            log_probs = (-0.5 * u.pow(2) - 0.5 * np.log(2 * np.pi)).sum()
            log_jacob = log_jacob.sum()
            loss = -(log_probs + log_jacob)

            if size_average:
                loss /= u.size(0)
            return loss

        # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50], gamma=0.1)
        if self.config.data.dataset == 'ImageNet':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.config.training.maximum_steps, eta_min=0.)
        else:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.config.training.n_epochs, eta_min=0.)

        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.run, 'logs', self.args.doc, 'checkpoint.pth'),
                                map_location=self.config.device)

            net.load_state_dict(states[0])
            optimizer.load_state_dict(states[1])
            begin_epoch = states[2]
            step = states[3]
            scheduler.load_state_dict(states[4])
        else:
            step = 0
            begin_epoch = 0

        # Train the model

        for epoch in range(begin_epoch, self.config.training.n_epochs):
            if self.config.data.dataset != 'ImageNet':
                scheduler.step()
            for batch_idx, (data, _) in enumerate(dataloader):
                net.train()
                # Transform to logit space since pixel values ranging from 0-1
                data = data.to(self.config.device) * 255. / 256.
                data += torch.rand_like(data) / 256.
                data = self.logit_transform(data)

                log_det_logit = F.softplus(-data).sum() + F.softplus(data).sum() + np.prod(
                    data.shape) * np.log(1 - 2 * self.config.data.lambda_logit)

                output, log_det = net(data)

                loss = flow_loss(output, log_det)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()

                # TODO: remove the below sanity check
                if loss.item() > 1e7:
                    return 0

                optimizer.step()
                ema_helper.update(net)

                bpd = (loss.item() * data.shape[0] - log_det_logit) / (np.log(2) * np.prod(data.shape)) + 8

                # validation
                # Do EMA

                if step % self.config.training.log_interval == 0:
                    net_test = ema_helper.ema_copy(net)
                    net_test.eval()
                    with torch.no_grad():
                        try:
                            test_data, _ = next(test_iter)
                        except StopIteration:
                            test_iter = iter(test_loader)
                            test_data, _ = next(test_iter)

                        test_data = test_data.to(self.config.device) * 255. / 256.
                        test_data += torch.rand_like(test_data) / 256.
                        test_data = self.logit_transform(test_data)

                        test_log_det_logit = F.softplus(-test_data).sum() + F.softplus(test_data).sum() + np.prod(
                            test_data.shape) * np.log(1 - 2 * self.config.data.lambda_logit)

                        test_output, test_log_det = net_test(test_data)
                        test_loss = flow_loss(test_output, test_log_det)
                        test_bpd = (test_loss.item() * test_data.shape[0] - test_log_det_logit) * (
                                1 / (np.log(2) * np.prod(test_data.shape))) + 8

                    tb_logger.add_scalar('training_loss', loss, global_step=step)
                    tb_logger.add_scalar('training_bpd', bpd, global_step=step)
                    tb_logger.add_scalar('test_loss', test_loss, global_step=step)
                    tb_logger.add_scalar('test_bpd', test_bpd, global_step=step)

                    logging.info(
                        "epoch: {}, batch: {}, training_loss: {}, test_loss: {}".format(epoch, batch_idx, loss.item(),
                                                                                        test_loss.item()))
                step += 1

                if self.config.data.dataset == 'ImageNet':
                    scheduler.step()
                    if step % self.config.training.snapshot_interval == 0:
                        states = [
                            net.state_dict(),
                            optimizer.state_dict(),
                            epoch + 1,
                            step,
                            scheduler.state_dict(),
                            ema_helper.state_dict()
                        ]
                        torch.save(states, os.path.join(self.args.run, 'logs', self.args.doc,
                                                        'checkpoint_batch_{}.pth'.format(step)))
                        torch.save(states, os.path.join(self.args.run, 'logs', self.args.doc, 'checkpoint.pth'))

                if step == self.config.training.maximum_steps:
                    states = [
                        net.state_dict(),
                        optimizer.state_dict(),
                        epoch + 1,
                        step,
                        scheduler.state_dict(),
                        ema_helper.state_dict()
                    ]
                    torch.save(states, os.path.join(self.args.run, 'logs', self.args.doc,
                                                    'checkpoint_last_batch.pth'))
                    torch.save(states, os.path.join(self.args.run, 'logs', self.args.doc, 'checkpoint.pth'))

                    return 0

            if self.config.data.dataset != 'ImageNet' and (epoch + 1) % self.config.training.snapshot_interval == 0:
                states = [
                    net.state_dict(),
                    optimizer.state_dict(),
                    epoch + 1,
                    step,
                    scheduler.state_dict(),
                    ema_helper.state_dict()
                ]
                torch.save(states, os.path.join(self.args.run, 'logs', self.args.doc,
                                                'checkpoint_epoch_{}.pth'.format(epoch + 1)))
                torch.save(states, os.path.join(self.args.run, 'logs', self.args.doc, 'checkpoint.pth'))

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
        import time
        torch.cuda.synchronize()
        start_time = time.time()

        transform = transforms.Compose([
            transforms.Resize(self.config.data.image_size),
            transforms.ToTensor()
        ])

        if self.config.data.dataset == 'CIFAR10':
            test_dataset = CIFAR10(os.path.join(self.args.run, 'datasets', 'cifar10'), train=False, download=True,
                                   transform=transform)
        elif self.config.data.dataset == 'MNIST':
            test_dataset = MNIST(os.path.join(self.args.run, 'datasets', 'fmnist'), train=False, download=True,
                                 transform=transform)

        elif self.config.data.dataset == 'ImageNet':
            test_dataset = OordImageNet('/atlas/u/yangsong/datasets/oord_imagenet', train=False, transform=transform)
                #ImageNet('/atlas/u/yangsong/datasets/imagenet', train=False, transform=transform)

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

        test_loader = DataLoader(test_dataset, batch_size=self.config.training.batch_size, shuffle=True,
                                 num_workers=4, drop_last=False)

        net = Net(self.config).to(self.config.device)
        net = DataParallelWithSampling(net)
        ema_helper = EMAHelper(mu=0.999)
        optimizer = self.get_optimizer(net.parameters())

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
        ema_helper.load_state_dict(states[5])
        ema_helper.ema(net)

        logging.info(
            "Loading the model from epoch {}".format(loaded_epoch))

        # Test the model
        net.eval()
        total_loss = 0
        total_bpd = 0
        total_n_data = 0

        logging.info("Generating samples")
        ## samples = []
        ## for i in range(4):
        ##     z = torch.randn(100, self.config.data.channels * self.config.data.image_size * self.config.data.image_size,
        ##                     device=self.config.device)
        ##     samples_temp = net.sampling(z)
        ##     samples_temp = self.sigmoid_transform(samples_temp)
        ##     samples.append(samples_temp)
        ## samples = torch.cat(samples, dim=0)

        z = torch.randn(64, self.config.data.channels * self.config.data.image_size * self.config.data.image_size,
                       device=self.config.device)
        samples = net.sampling(z)
        samples = self.sigmoid_transform(samples)

        samples = make_grid(samples, 8)
        save_image(samples, 'samples_cifar10_30.png')

        logging.info("Calculating overall bpd")

        with torch.no_grad():
            for batch_idx, (test_data, _) in enumerate(tqdm.tqdm(test_loader)):
                test_data = test_data.to(self.config.device) * 255. / 256.
                test_data += torch.rand_like(test_data) / 256.
                test_data = self.logit_transform(test_data)

                test_log_det_logit = F.softplus(-test_data).sum() + F.softplus(test_data).sum() + np.prod(
                    test_data.shape) * np.log(1 - 2 * self.config.data.lambda_logit)

                test_output, test_log_det = net(test_data)
                test_loss = flow_loss(test_output, test_log_det)

                test_bpd = (test_loss.item() * test_data.shape[0] - test_log_det_logit) * (
                        1 / (np.log(2) * np.prod(test_data.shape))) + 8

                total_loss += test_loss * test_data.shape[0]
                total_bpd += test_bpd * test_data.shape[0]
                total_n_data += test_data.shape[0]
        logging.info(
            "Total batch:{}\nTotal loss: {}\nTotal bpd: {}".format(batch_idx + 1, total_loss.item() / total_n_data,
                                                                   total_bpd.item() / total_n_data))

        torch.cuda.synchronize()
        time_taken = time.time() - start_time
        print("Run-Time: %.4f s" % time_taken)


    def invert_experiment(self):
        transform = transforms.Compose([
            transforms.Resize(self.config.data.image_size),
            transforms.ToTensor()
        ])

        if self.config.data.dataset == 'CIFAR10':
            dataset = CIFAR10(os.path.join(self.args.run, 'datasets', 'cifar10'), train=True, download=True,
                              transform=transform)

        elif self.config.data.dataset == 'MNIST':
            dataset = MNIST(os.path.join(self.args.run, 'datasets', 'mnist'), train=True, download=True,
                            transform=transform)

        elif self.config.data.dataset == 'ImageNet':
            dataset = OordImageNet('/atlas/u/yangsong/datasets/oord_imagenet', train=False, transform=transform)


        dataloader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=4,
                                drop_last=True)

        net = Net(self.config).to(self.config.device)
        net = DataParallelWithSampling(net)
        states = torch.load(os.path.join(self.args.run, 'logs', self.args.doc, 'checkpoint.pth'),
                            map_location=self.config.device)

        state_dict = {}
        layer_stop = 0
        key_set = states[0].keys()
        for k in key_set:
            string = 'module.layers.'
            start = len(string)
            key = k[start:]
            end = key.find(".")
            index = int(key[:end])
            if index <= layer_stop:
                state_dict[k] = states[0][k]


        net.load_state_dict(states[0])
        loaded_epoch = states[2]

        logging.info(
            "Loading the model from epoch {}".format(loaded_epoch))

        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(dataloader):
                if batch_idx < 3:
                    continue
                data = data.to(self.config.device)
                data_img = make_grid(data, 10)
                save_image(data_img, 'ground_truth_imagenet_n.png')
                data = data * 255. / 256.
                noise = torch.rand_like(data) / 256.
                data += noise
                data = self.logit_transform(data)
                output, log_det = net.forward(data)
                samples = net.sampling(output)
                samples = self.sigmoid_transform(samples)
                samples -= noise
                samples = samples * 256. / 255.
                samples = make_grid(samples, 10)
                save_image(samples, 'invert_results_imagenet_n.png')
                break

        # only inverse the last layer

    def newton_analysis(self):
        transform = transforms.Compose([
            transforms.Resize(self.config.data.image_size),
            transforms.ToTensor()
        ])
        if self.config.data.dataset == 'CIFAR10':
            dataset = CIFAR10(os.path.join(self.args.run, 'datasets', 'cifar10'), train=True, download=True,
                              transform=transform)

        elif self.config.data.dataset == 'MNIST':
            dataset = MNIST(os.path.join(self.args.run, 'datasets', 'mnist'), train=True, download=True,
                            transform=transform)

        elif self.config.data.dataset == 'ImageNet':
            dataset = OordImageNet('/atlas/u/yangsong/datasets/oord_imagenet', train=False, transform=transform)

        dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4,
                                drop_last=True)

        net = Net(self.config).to(self.config.device)
        net = DataParallelWithSampling(net)
        optimizer = self.get_optimizer(net.parameters())
        states = torch.load(os.path.join(self.args.run, 'logs', self.args.doc, 'checkpoint.pth'),
                            map_location=self.config.device)

        net.load_state_dict(states[0])
        optimizer.load_state_dict(states[1])
        loaded_epoch = states[2]
        logging.info(
            "Loading the model from epoch {}".format(loaded_epoch))

        # for batch_idx, (data, _) in enumerate(dataloader):
        #     if batch_idx == 0:
        #         continue
        #     data = data.to(self.config.device) * 255. / 256.
        #     data += torch.rand_like(data) / 256.
        #     data = self.logit_transform(data)
        #     break

        val = []
        y1 = []
        y2 = []
        results = {}
        #output, log_det = net(data)
        newton_iter = self.config.analysis.newton_iter
        standard_deviation = []
        # for lr in np.arange(self.config.analysis.lower_bound, self.config.analysis.upper_bound,
        #                       self.config.analysis.interval):
            #print("lr ", lr)
            #self.config.analysis.newton_lr = lr

        for i in range(newton_iter):
            with torch.no_grad():
                if i % 10 != 0:
                    continue
                self.config.model.n_iters = i + 1
                print('iteration: {}'.format(self.config.model.n_iters))
                diff_array = []
                for b in range(4):
                    for batch_idx, (data, _) in enumerate(dataloader):
                        if batch_idx != b:
                            continue
                        data = data.to(self.config.device) * 255. / 256.
                        data += torch.rand_like(data) / 256.
                        data = self.logit_transform(data)
                        break
                    output, log_det = net(data)
                    inverse = net.sampling(output)
                    diff = (inverse - data).view(data.shape[0], -1)
                    diff_array.append(diff)

                diff = torch.cat(diff_array, dim=0)
                size = np.prod(data.shape)
                l2 = torch.log(torch.sqrt((diff.pow(2).sum(dim=-1))) / diff.shape[-1])
                std = l2.std(dim=0) #/ diff.shape[-1]
                diff = l2.mean(dim=0)#l2.sum(dim=0) #/ size

                #print(diff.data)
                val.append(np.exp(diff.item()))
                y1.append(np.exp(diff.item() - std.item()))
                y2.append(np.exp(diff.item() + std.item()))
                standard_deviation.append(std.item())
                #if diff.data < 1e-6:
                    #results[lr] = [diff.data, i]
                    #break
        #print(results)
        #return

        sns.set(style='darkgrid')
        x = np.arange(0, newton_iter, 10) + 1
        plt.xticks(np.arange(min(x) - 1, max(x) + 1, 50)) #100))
        plt.yscale('log')
        #plt.yscale('symlog')
        mnist, = plt.plot(x, val, label='MNIST', linewidth=2.)
        #print(val)
        plt.fill_between(x, y1, y2, alpha=0.35)
        #plt.legend(handles=[mnist])
        #print(standard_deviation)
        with open("MNIST_val", 'wb') as pfile:
            pickle.dump(val, pfile, protocol=pickle.HIGHEST_PROTOCOL)

        with open("MNIST_y1", 'wb') as pfile:
            pickle.dump(y1, pfile, protocol=pickle.HIGHEST_PROTOCOL)

        with open("MNIST_y2", 'wb') as pfile:
            pickle.dump(y2, pfile, protocol=pickle.HIGHEST_PROTOCOL)

        # plot cifar10
        self.config.model.n_layers = 21
        self.config.model.latent_size = 85
        self.config.data.dataset = "CIFAR10"
        self.config.data.image_size = 32
        self.config.data.channels = 3
        self.config.data.lambda_logit = 0.05
        self.config.analysis.newton_lr = 1.1

        train_transform = transforms.Compose([
            transforms.Resize(self.config.data.image_size),
            transforms.ToTensor()
        ])

        dataset = CIFAR10(os.path.join(self.args.run, 'datasets', 'cifar10'), train=True, download=True,
                          transform=train_transform)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4,
                                drop_last=True)

        net = Net(self.config).to(self.config.device)
        net = DataParallelWithSampling(net)
        optimizer = self.get_optimizer(net.parameters())
        states = torch.load(os.path.join(self.args.run, 'logs', 'density_cifar10_21x85_nozeroinit', 'checkpoint.pth'),
                            map_location=self.config.device)

        net.load_state_dict(states[0])
        optimizer.load_state_dict(states[1])
        loaded_epoch = states[2]
        logging.info(
            "Loading the model from epoch {}".format(loaded_epoch))

        val = []
        y1 = []
        y2 = []
        newton_iter = self.config.analysis.newton_iter
        standard_deviation = []
        for i in range(newton_iter):
            with torch.no_grad():
                if i % 10 != 0:
                    continue
                self.config.model.n_iters = i + 1
                print('iteration: {}'.format(self.config.model.n_iters))
                diff_array = []
                for b in range(4):
                    for batch_idx, (data, _) in enumerate(dataloader):
                        if batch_idx != b:
                            continue
                        data = data.to(self.config.device) * 255. / 256.
                        data += torch.rand_like(data) / 256.
                        data = self.logit_transform(data)
                        break
                    output, log_det = net(data)
                    inverse = net.sampling(output)
                    diff = (inverse - data).view(data.shape[0], -1)
                    diff_array.append(diff)

                diff = torch.cat(diff_array, dim=0)
                l2 = torch.log(torch.sqrt((diff.pow(2).sum(dim=-1))) / diff.shape[-1])
                std = l2.std(dim=0)
                diff = l2.mean(dim=0)

                print(diff.data)
                val.append(np.exp(diff.item()))
                y1.append(np.exp(diff.item() - std.item()))
                y2.append(np.exp(diff.item() + std.item()))
                standard_deviation.append(std.item())

        # for batch_idx, (data, _) in enumerate(dataloader):
        #     if batch_idx == 0:
        #         continue
        #     data = data.to(self.config.device) * 255. / 256.
        #     data += torch.rand_like(data) / 256.
        #     data = self.logit_transform(data)
        #     break
        #
        # val = []
        # output, log_det = net(data)
        # newton_iter = self.config.analysis.newton_iter
        # for i in range(newton_iter):
        #     with torch.no_grad():
        #         if i % 10 != 0:
        #             continue
        #         self.config.model.n_iters = i + 1
        #         print('iteration: {}'.format(self.config.model.n_iters))
        #         inverse = net.sampling(output)
        #         reconstruct_output, _ = net.forward(inverse)
        #         size = np.prod(reconstruct_output.shape)
        #         diff = torch.sqrt(torch.sum((reconstruct_output - output).pow(2))) / size
        #         print(diff.data)
        #         val.append(diff.data)

        x = np.arange(0, newton_iter, 10) + 1
        plt.xticks(np.arange(min(x) - 1, max(x) + 1, 50))
        plt.yscale('log')
        cifar10, = plt.plot(x, val, label='CIFAR-10', linewidth=2.)
        plt.fill_between(x, y1, y2, alpha=0.35)
        with open("CIFAR10_val", 'wb') as pfile:
            pickle.dump(val, pfile, protocol=pickle.HIGHEST_PROTOCOL)
        with open("CIFAR10_y1", 'wb') as pfile:
            pickle.dump(y1, pfile, protocol=pickle.HIGHEST_PROTOCOL)
        with open("CIFAR10_y2", 'wb') as pfile:
            pickle.dump(y2, pfile, protocol=pickle.HIGHEST_PROTOCOL)
        #plt.legend(handles=[cifar10])

        # plot imagenet
        self.config.model.n_layers = 21
        self.config.model.latent_size = 85
        self.config.data.dataset = "ImageNet"
        self.config.data.image_size = 32
        self.config.data.channels = 3
        self.config.data.lambda_logit = 0.05
        self.config.analysis.newton_lr = 1.15

        train_transform = transforms.Compose([
            transforms.Resize(self.config.data.image_size),
            transforms.ToTensor()
        ])

        dataset = OordImageNet('/atlas/u/yangsong/datasets/oord_imagenet', train=False,
                                         transform=train_transform)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4,
                                drop_last=True)

        net = Net(self.config).to(self.config.device)
        net = DataParallelWithSampling(net)
        optimizer = self.get_optimizer(net.parameters())
        states = torch.load(os.path.join(self.args.run, 'logs', 'copy_imagenet', 'checkpoint.pth'),
                            map_location=self.config.device)

        net.load_state_dict(states[0])
        optimizer.load_state_dict(states[1])
        loaded_epoch = states[2]
        logging.info(
            "Loading the model from epoch {}".format(loaded_epoch))

        val = []
        y1 = []
        y2 = []
        newton_iter = self.config.analysis.newton_iter
        standard_deviation = []

        for i in range(newton_iter):
            with torch.no_grad():
                if i % 10 != 0:
                    continue
                self.config.model.n_iters = i + 1
                print('iteration: {}'.format(self.config.model.n_iters))
                diff_array = []

                for b in range(4):
                    for batch_idx, (data, _) in enumerate(dataloader):
                        if batch_idx != b:
                            continue
                        data = data.to(self.config.device) * 255. / 256.
                        data += torch.rand_like(data) / 256.
                        data = self.logit_transform(data)
                        break
                    output, log_det = net(data)
                    inverse = net.sampling(output)
                    diff = (inverse - data).view(data.shape[0], -1)
                    diff_array.append(diff)

                diff = torch.cat(diff_array, dim=0)
                l2 = torch.log(torch.sqrt((diff.pow(2).sum(dim=-1))) / diff.shape[-1])
                std = l2.std(dim=0)
                diff = l2.mean(dim=0)

                #print(diff.data)
                val.append(np.exp(diff.item()))
                y1.append(np.exp(diff.item() - std.item()))
                y2.append(np.exp(diff.item() + std.item()))
                standard_deviation.append(std.item())



        # for batch_idx, (data, _) in enumerate(dataloader):
        #     if batch_idx == 0:
        #         continue
        #     data = data.to(self.config.device) * 255. / 256.
        #     data += torch.rand_like(data) / 256.
        #     data = self.logit_transform(data)
        #     break
        #
        # val = []
        # output, log_det = net(data)
        # newton_iter = self.config.analysis.newton_iter
        # for i in range(newton_iter):
        #     with torch.no_grad():
        #         if i % 10 != 0:
        #             continue
        #         self.config.model.n_iters = i + 1
        #         print('iteration: {}'.format(self.config.model.n_iters))
        #         inverse = net.sampling(output)
        #         reconstruct_output, _ = net.forward(inverse)
        #         size = np.prod(reconstruct_output.shape)
        #         diff = torch.sqrt(torch.sum((reconstruct_output - output).pow(2))) / size
        #         print(diff.data)
        #         val.append(diff.data)

        x = np.arange(0, newton_iter, 10) + 1
        plt.xticks(np.arange(min(x) - 1, max(x) + 1, 50))
        plt.yscale('log')
        imagenet, = plt.plot(x, val, label=r'ImageNet 32$\times$32', linewidth=2.)
        plt.fill_between(x, y1, y2, alpha=0.35)
        with open("Imagenet_val", 'wb') as pfile:
            pickle.dump(val, pfile, protocol=pickle.HIGHEST_PROTOCOL)
        with open("Imagenet_y1", 'wb') as pfile:
            pickle.dump(y1, pfile, protocol=pickle.HIGHEST_PROTOCOL)
        with open("Imagenet_y2", 'wb') as pfile:
            pickle.dump(y2, pfile, protocol=pickle.HIGHEST_PROTOCOL)
        #plt.legend(handles=[imagenet])


        plt.xlabel("Number of Iterations")
        plt.ylabel(r'Normalized  $L_2$ Rec. Error')  # "Normalized L2 Rec. Error")
        plt.legend(handles=[mnist, cifar10, imagenet])
        plt.savefig('newton_analysis')



    def interpolation(self):
        transform = transforms.Compose([
            transforms.Resize(self.config.data.image_size),
            transforms.ToTensor()
        ])

        if self.config.data.dataset == 'CIFAR10':
            dataset = CIFAR10(os.path.join(self.args.run, 'datasets', 'cifar10'), train=True, download=True,
                              transform=transform)
        elif self.config.data.dataset == 'MNIST':
            dataset = MNIST(os.path.join(self.args.run, 'datasets', 'mnist'), train=True, download=True,
                            transform=transform)

        elif self.config.data.dataset == 'ImageNet':
            dataset = OordImageNet('/atlas/u/yangsong/datasets/oord_imagenet', train=False, transform=transform)

        dataloader = DataLoader(dataset, batch_size=50, shuffle=True, num_workers=4,
                                drop_last=True)

        net = Net(self.config).to(self.config.device)
        net = DataParallelWithSampling(net)
        optimizer = self.get_optimizer(net.parameters())
        states = torch.load(os.path.join(self.args.run, 'logs', self.args.doc, 'checkpoint.pth'),
                            map_location=self.config.device)
        net.load_state_dict(states[0])
        optimizer.load_state_dict(states[1])
        loaded_epoch = states[2]
        logging.info(
            "Loading the model from epoch {}".format(loaded_epoch))

        def linear_interpolation(x1, x2, x3, x4, phi, phi_p, net):
            with torch.no_grad():
                z1, _ = net.forward(x1)
                z2, _ = net.forward(x2)
                z3, _ = net.forward(x3)
                z4, _ = net.forward(x4)
                z = math.cos(phi) * (math.cos(phi_p) * z1 + math.sin(phi_p) * z2) + \
                    math.sin(phi) * (math.cos(phi_p) * z3 + math.sin(phi_p) * z4)
                x = net.sampling(z)
                x = self.sigmoid_transform(x)
            return x


        for batch_idx, (data, _) in enumerate(dataloader):
            data = data.to(self.config.device)
            images = torch.zeros((64, *data[0].shape))
            data = data * 255. / 256.
            data += torch.rand_like(data) / 256.
            data = self.logit_transform(data)
            x1 = data[3].unsqueeze(dim=0)
            x2 = data[15].unsqueeze(dim=0)
            x3 = data[32].unsqueeze(dim=0)
            x4 = data[17].unsqueeze(dim=0)

            for i in range(8):
                for j in range(8):
                    phi = math.pi * i / 14.
                    phi_p = math.pi * j / 14.
                    xt = linear_interpolation(x1, x2, x3, x4, phi, phi_p, net)
                    images[i*8+j] = xt.squeeze(dim=0)
            images = make_grid(images, 8)
            save_image(images, 'interpolation.png')
            break


