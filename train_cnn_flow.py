import os
import argparse
import pickle
from model_flow import *

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device('cuda')
# device = torch.device('cpu')

n_epochs = 300
batch_size_train = 64
batch_size_test = 64  # 1000
learning_rate = 1e-2#1.5e-3  # 5e-7
momentum = 0.9
log_interval = 10

#lambda_logit = 1e-6 #for MNIST
lambda_logit = 0.05

'''
train_loader = torch.utils.data.DataLoader(
 torchvision.datasets.MNIST('./files/', train=True, download=True,
                            transform=torchvision.transforms.Compose([
                              torchvision.transforms.ToTensor()])),
 batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
torchvision.datasets.MNIST('./files/', train=False, download=True,
                            transform=torchvision.transforms.Compose([
                              torchvision.transforms.ToTensor()])),
 batch_size=batch_size_test, shuffle=True)
'''


train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.CIFAR10('./dataset/', train=True, download=True,
                                 transform=torchvision.transforms.Compose([
                                     torchvision.transforms.ToTensor()#, 
                                 ])),
    batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.CIFAR10('./dataset/', train=False, download=True,
                                 transform=torchvision.transforms.Compose([
                                     torchvision.transforms.ToTensor()#, 
                                 ])),
    batch_size=batch_size_test, shuffle=True)


#net = Net(BasicBlockA, BasicBlockB, [4, 4, 1, 1], [1, 1, 1, 1], image_size=28, input_channel=1).to(device)
net = Net(BasicBlockA, BasicBlockB, [4, 4, 1, 1], [1, 1, 1, 1], image_size=32, input_channel=3).to(device)
optimizer = optim.Adam(net.parameters(), weight_decay=1e-6, lr=learning_rate)

stats = {}
stats['train_loss'] = []
stats['train_accuracy'] = []
stats['val_loss'] = []
stats['val_accuracy'] = []
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50)

def flow_loss(u, log_jacob, size_average=True):
    log_probs = (-0.5 * u.pow(2) - 0.5 * math.log(2 * math.pi)).sum()
    loss = -(log_probs + log_jacob)
  
    if size_average:
        loss /= u.size(0)
    return loss

# Train the model
for _ in range(n_epochs):
    # gradually change learning rate
    lr_scheduler.step()
    print('Now processing epoch {}'.format(_))
    train_loss = 0
    total_train = 0

    train_count = 0 #number of train samples 
    net.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if data.shape[0] != batch_size_train: continue

        # Transform to logit space since pixel values ranging from 0-1
        train_count += data.shape[0]
        data = lambda_logit + (1 - 2 * lambda_logit) * data # Logit transform lambda + (1 - 2 * lambda) * x
        log_det_logit = -(-(data.log()).sum() - ((1-data).log()).sum() + np.prod(data.shape) * np.log(1 - 2 * lambda_logit))
        data = np.log(data) - np.log(1-data)
        data, target = data.to(device), target.to(device)
        output, log_det = net(data)
        #log_det += log_det_logit
        loss = flow_loss(output, log_det)
        train_loss += loss.item() 
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()  
    stats['train_loss'].append(train_loss)
    print('Average Training Loss: {}. \n'.format(train_loss * np.log2(math.e) * batch_size_train/train_count))

    # Test
    net.eval()
    test_loss = 0
    test_count = 0 #number of test samples 
    with torch.no_grad():
        for data, target in test_loader:
            if data.shape[0] != batch_size_train: continue
            # transform to logit space
            test_count += data.shape[0]
            data = lambda_logit + (1 - 2 * lambda_logit) * data # Logit transform lambda + (1 - 2 * lambda) * x
            # make sure sign is correct
            log_det_logit = -(-(data.log()).sum() - ((1-data).log()).sum() + np.prod(data.shape) * np.log(1 - 2 * lambda_logit))
            
            data = np.log(data/(1-data))
            data, target = data.to(device), target.to(device)
            output, log_det = net(data)
            #log_det += log_det_logit
            loss = flow_loss(output, log_det, False).item()
            test_loss += loss
        test_loss /= (test_count)
        stats['val_loss'].append(test_loss)
        print('\nTest set: loss: {:.4f}\n'.format(test_loss * np.log2(math.e)))
        
    with open('./train_stats', 'wb') as f:
        pickle.dump(stats, f, protocol=pickle.HIGHEST_PROTOCOL)
    
