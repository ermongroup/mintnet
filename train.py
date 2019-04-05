import os
import argparse
import pickle
from updated_model import *

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device('cuda')
# device = torch.device('cpu')

n_epochs = 300
batch_size_train = 64
batch_size_test = 64  
learning_rate = 1e-3 
momentum = 0.9

train_loader = torch.utils.data.DataLoader(
torchvision.datasets.MNIST('./files/', train=True, download=True,
                            transform=torchvision.transforms.Compose([
                              torchvision.transforms.ToTensor(),
                            ])),
batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
torchvision.datasets.MNIST('./files/', train=False, download=True,
                            transform=torchvision.transforms.Compose([
                              torchvision.transforms.ToTensor(),
                            ])),
batch_size=batch_size_test, shuffle=True)

net = Net(BasicBlockA, BasicBlockB, [2, 2, 1, 1], [2, 2, 1, 1], image_size=28, input_channel=1).to(device)

'''
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.CIFAR10('./dataset/', train=True, download=True,
                                 transform=torchvision.transforms.Compose([
                                     torchvision.transforms.ToTensor(),                                     
                                 ])),
    batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.CIFAR10('./dataset/', train=False, download=True,
                                 transform=torchvision.transforms.Compose([
                                     torchvision.transforms.ToTensor(),
                                 ])),
    batch_size=batch_size_test, shuffle=True)

net = Net(BasicBlockA, BasicBlockB, [4, 4, 1, 1], [32, 32, 1, 1], image_size=32, input_channel=3).to(device)
'''


optimizer = optim.Adam(net.parameters(), weight_decay=1e-4, lr=learning_rate)

stats = {}
stats['train_loss'] = []
stats['train_accuracy'] = []
stats['val_loss'] = []
stats['val_accuracy'] = []
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=1 / 1.5)

for _ in range(n_epochs):
    # gradually change learning rate
    lr_scheduler.step()

    print('Now processing epoch {}'.format(_))
    train_loss = 0
    total_train = 0
    train_correct = 0

    net.train()
    for batch_idx, (data, target) in enumerate(train_loader):       
        if data.shape[0] != batch_size_train: continue
     
        data, target = data.to(device), target.to(device)
        output = net(data)
        loss = F.nll_loss(output, target)
        pred = output.data.max(1, keepdim=True)[1]
        train_correct += pred.eq(target.data.view_as(pred)).sum()
        total_train += target.shape[0]
        train_loss += loss.item()
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
    train_loss = train_loss * batch_size_train / len(train_loader.dataset)
    stats['train_loss'].append(train_loss)
    stats['train_accuracy'].append((100. * train_correct / total_train).cpu().numpy())
    print('Average Training Loss: {}\nAccuracy: {}%'.format(train_loss, 100. * train_correct / total_train))

    # Test
    net.eval()
    test_loss = 0
    total_test = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if data.shape[0] != batch_size_train: continue            
            total_test += data.shape[0]            
            data, target = data.to(device), target.to(device)
            output = net(data)
            
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
            
        test_loss /= total_test
        stats['val_loss'].append(test_loss)
        stats['val_accuracy'].append((100. * correct / len(test_loader.dataset)).cpu().numpy())
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, total_test,
            100. * correct / total_test))
    
    with open('./train_stats', 'wb') as f:
        pickle.dump(stats, f, protocol=pickle.HIGHEST_PROTOCOL)
