import os
import argparse
import pickle
import math
from model_flow import *

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device('cuda')
# device = torch.device('cpu')

torch.manual_seed(0)
np.random.seed(0)

n_epochs = 500
batch_size_train = 64
batch_size_test = 64  # 1000
learning_rate = 1.5e-3  # 5e-7
momentum = 0.9
log_interval = 10

#lambda_logit = 1e-6 #for MNIST
lambda_logit = 0.05


train_loader = torch.utils.data.DataLoader(
 torchvision.datasets.MNIST('./mnist/', train=True, download=True,
                            transform=torchvision.transforms.Compose([
                              torchvision.transforms.ToTensor()])),
                            #transform=torchvision.transforms.Compose([transforms.ToPILImage()])),
 batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
torchvision.datasets.MNIST('./mnist/', train=False, download=True,
                            transform=torchvision.transforms.Compose([
                              torchvision.transforms.ToTensor()])),
                           #transform=torchvision.transforms.Compose([transforms.ToPILImage()])),
 batch_size=batch_size_test, shuffle=True)

#[2, 2, 1, 1], [4, 1, 1, 1]
net = Net(BasicBlockA, BasicBlockB, [2, 2, 1, 1], [32, 16, 8, 1], image_size=28, input_channel=1).to(device)


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
    
net = Net(BasicBlockA, BasicBlockB, [4, 4, 1, 1], [1, 1, 1, 1], image_size=32, input_channel=3).to(device)
'''


optimizer = optim.Adam(net.parameters(), weight_decay=1e-8, lr=learning_rate)

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

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Train the model
for _ in range(n_epochs):
    # gradually change learning rate
    if _ == 10:
        learning_rate /= 5
    if _ == 20:
        learning_rate /= 2
    
    #only later added!!
    if _ == 21:
        learning_rate /= 10
        
        
    lr_scheduler.step()
    print('Now processing epoch {}'.format(_))
    train_loss = 0
    total_train = 0
    batch_num = 0
    train_count = 0 #number of train samples 
    net.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        #if train_count > 6000: break
        batch_num += 1
        if data.shape[0] != batch_size_train: continue

        # Transform to logit space since pixel values ranging from 0-1
        train_count += data.shape[0]
        data = lambda_logit + (1 - 2 * lambda_logit) * data # Logit transform lambda + (1 - 2 * lambda) * x
   
        data = np.log(data) - np.log(1-data)
        sigmoid_data = sigmoid(data)
        log_det_logit = -(sigmoid_data.log()).sum() - ((1-sigmoid_data).log()).sum() + np.prod(data.shape) * np.log(1 - 2 * lambda_logit)
        data, target = data.to(device), target.to(device)
        output, log_det = net(data)
        #log_det += log_det_logit
        loss = flow_loss(output, log_det)
        #train_loss += loss.item() 
        train_loss += (loss.item() * batch_size_train - log_det_logit) * (1/(np.log(2)*np.prod(data.shape))) 
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()  
    stats['train_loss'].append(train_loss)
    print('Average Training Loss: {}. \n'.format(train_loss/batch_num + 8))

    # Test
    net.eval()
    test_loss = 0
    test_count = 0 #number of test samples 
    batch_test = 0
    with torch.no_grad():
        for data, target in test_loader:
            #if test_count != 0: break
            if data.shape[0] != batch_size_train: continue
            batch_test += 1
            # transform to logit space
            test_count += data.shape[0]
            data = lambda_logit + (1 - 2 * lambda_logit) * data # Logit transform lambda + (1 - 2 * lambda) * x
            # make sure sign is correct 
            data = np.log(data)- np.log(1-data)
            sigmoid_data = sigmoid(data)
            log_det_logit = -(sigmoid_data.log()).sum() - ((1-sigmoid_data).log()).sum() + np.prod(data.shape) * np.log(1 - 2 * lambda_logit)
            data, target = data.to(device), target.to(device)
            output, log_det = net(data)
            #log_det += log_det_logit
            loss = flow_loss(output, log_det, False).item()
            #test_loss += loss
            test_loss += (loss - log_det_logit) * (1/(np.log(2)*np.prod(data.shape))) 
            
        #test_loss /= (test_count)
        stats['val_loss'].append(test_loss)
        print('\nTest set: loss: {:.4f}\n'.format(test_loss/batch_test + 8))
        
    with open('./train_stats', 'wb') as f:
        pickle.dump(stats, f, protocol=pickle.HIGHEST_PROTOCOL)
    
