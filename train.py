import os
import argparse
import pickle
#from model_architecture import *
from updated_model import *

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device('cuda')

#device = torch.device('cpu')
n_epochs = 300
batch_size_train = 64
batch_size_test = 64#1000
learning_rate = 1e-3#5e-7
momentum = 0.9
log_interval = 10

#MNIST
#train_loader = torch.utils.data.DataLoader(
# torchvision.datasets.MNIST('./files/', train=True, download=True,
#                            transform=torchvision.transforms.Compose([
#                              torchvision.transforms.ToTensor(),
#                              torchvision.transforms.Normalize(
#                                (0.1307,), (0.3081,))
#                            ])),
# batch_size=batch_size_train, shuffle=True)

#test_loader = torch.utils.data.DataLoader(
# torchvision.datasets.MNIST('./files/', train=False, download=True,
#                            transform=torchvision.transforms.Compose([
#                              torchvision.transforms.ToTensor(),
#                              torchvision.transforms.Normalize(
#                                (0.1307,), (0.3081,))
#                            ])),
# batch_size=batch_size_test, shuffle=True)

train_loader = torch.utils.data.DataLoader(
   torchvision.datasets.CIFAR10('./dataset/', train=True, download=True,
                              transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                  (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                              ])),
   batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.CIFAR10('./dataset/', train=False, download=True,
                              transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                  (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                              ])),
   batch_size=batch_size_test, shuffle=True)


net = Net(BasicBlockA, BasicBlockB,[4,4,1,1],[1,1,1,1],image_size=32, input_channel=3).to(device)
#net = DimIncrease(3, 3, stride=1, kernel=3).to(device)
#net = Net(BasicBlockA, BasicBlockB,[2,1,1,1], image_size=32,input_channel=3).to(device)
#optimizer = optim.SGD(net.parameters(), weight_decay=1e-4, lr=learning_rate,
                      #momentum=momentum)
optimizer = optim.Adam(net.parameters(), weight_decay=1e-4, lr=learning_rate)

# Train the model
net.train()
stats = {}

stats['train_loss'] = []
stats['train_accuracy'] = []
stats['val_loss'] = []
stats['val_accuracy'] = []
#train_loss = 0
for _ in range(n_epochs):
	#gradually change learning rate
	if _  % 50 ==0 :
		for g in optimizer.param_groups:
			g['lr'] /= 1.5
	#else:
	#	for g in optimizer.param_groups:
	#		g['lr'] -= 5e-7

	print('Now processing epoch {}'.format(_))
	train_loss = 0
	total_train = 0
	train_correct = 0

	for batch_idx, (data, target) in enumerate(train_loader):
		data, target = data.to(device), target.to(device)
		output = net.forward(data)
		#pdb.set_trace()
		loss = F.nll_loss(output, target)

		pred = output.data.max(1, keepdim=True)[1]
		train_correct += pred.eq(target.data.view_as(pred)).sum()
		total_train += target.shape[0]
		#pdb.set_trace()
		params = list(net.parameters())
		train_loss += loss.item()
		# Backward and optimize
		optimizer.zero_grad()
		loss.backward(retain_graph=True)
		optimizer.step()
		#print(net.conv2.bias.grad)
	#pdb.set_trace()
	train_loss = train_loss * batch_size_train/ len(train_loader.dataset)
	stats['train_loss'].append(train_loss)
	stats['train_accuracy'].append((100. * train_correct/total_train).cpu().numpy())
	print('Average Training Loss: {}\nAccuracy: {}%'.format(train_loss, 100. * train_correct/total_train))

	# Test
	net.eval()
	test_loss = 0
	correct = 0
	with torch.no_grad():
		for data, target in test_loader:
			data, target = data.to(device), target.to(device)
			output = net(data)
			test_loss += F.nll_loss(output, target, size_average=False).item()
			pred = output.data.max(1, keepdim=True)[1]
			correct += pred.eq(target.data.view_as(pred)).sum()
		test_loss /= len(test_loader.dataset)
		#test_losses.append(test_loss)
		#pdb.set_trace()
		stats['val_loss'].append(test_loss)
		stats['val_accuracy'].append((100. * correct / len(test_loader.dataset)).cpu().numpy())
		print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
	    test_loss, correct, len(test_loader.dataset),
	    100. * correct / len(test_loader.dataset)))		

	with open('./train_stats', 'wb') as f:
		pickle.dump(stats,f,protocol=pickle.HIGHEST_PROTOCOL)

