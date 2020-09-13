#!/home/ubuntu/anaconda3/bin/python3

import os

from tqdm import tqdm
from tensorboardX import SummaryWriter

import torch
import torch.optim as optim
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader

# own modules
import utils
import mobilenet, resnet, raspnet

# settings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: "+str(device))
model_type = 'small'
model_name = 'onet_a'
prev_class_num = 2
curr_class_num = 2

# age subdivision
custom_at = (50, 200)
#custom_at = (25, 50, 75, 200)

load_prev = False
prev_model = str(prev_class_num)+'.pt'
save_dir = './tensorlog'
learning_rate = 1e-2
opt = "Adam"
epochs = 60
unfreeze_epoch = 2
batch_size = 128
decay_step = [25, 45, 55]
decay_rate = 0.1
verb_step = 10
save_epoch = 5

if not os.path.exists(save_dir):
	os.makedirs(save_dir)
if model_type == 'large':
	resize_shape = (224, 224)
else:
	resize_shape = (48, 48)

writer = SummaryWriter(log_dir=save_dir)	# tensorboard writer


def main():
	# get data (remember to substitute path to UTKFace dataset)
	x_train, y_train, x_valid, y_valid = utils.get_images(r'../UTKFace', age_thresh=custom_at, resize_shape=resize_shape)

	# define reader
	transform_train = transforms.Compose([
		transforms.ToPILImage(),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
	])
	transform_valid = transforms.Compose([
		transforms.ToPILImage(),
		transforms.ToTensor(),
		transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
	])
	train_reader = DataLoader(utils.UTKDataLoader(x_train, y_train, tsfm=transform_train), batch_size=batch_size,
							  num_workers=4, shuffle=True)
	valid_reader = DataLoader(utils.UTKDataLoader(x_valid, y_valid, tsfm=transform_valid), batch_size=batch_size,
							  num_workers=4, shuffle=False)

	# network
	if model_type == 'large':
		if model_name == 'resnet':
			net = resnet.resnet34(True)
			net.fc = nn.Linear(512, prev_class_num)
		elif model_name == 'mobilenet':
			net = mobilenet.mobilenet_v2(True)
			net.classifier[1] = nn.Linear(1280, prev_class_num)
		else:
			raise NotImplementedError
	else:
		net = raspnet.raspnet(name=model_name, class_num=prev_class_num)
	writer.add_graph(net, torch.rand(1, 3, *resize_shape))

	if load_prev == True:
		# NOTE load previously trained model here
		checkpoint = torch.load(prev_model)
		net.load_state_dict(checkpoint["state_dict"])
		for param in net.parameters():	# freeze parameters
			param.requires_grad = False

	net.dense6_1 = nn.Linear(256, curr_class_num)
	#optimizer.load_state_dict(checkpoint["opt_dict"])

	net.to(device)
	net.train()	# set training mode

	# define loss function
	criterion = nn.CrossEntropyLoss().to(device)
	# set up optimizer
	if opt == 'Adam':
		optimizer = optim.Adam(net.parameters(), lr=learning_rate, amsgrad=True)
	else:
		optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
	scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=decay_step, gamma=decay_rate)

	"""
	# NOTE manually reset learning rate to desired value
	for g in optimizer.param_groups:
		g['lr'] = learning_rate
	"""

	# train
	for epoch in range(1, epochs+1):
		
		if load_prev:	# only do this when in curriculum training mode
			if epoch == unfreeze_epoch:
				print("### epoch: "+str(epoch)+" unfreezed parameters!")
				for param in net.parameters():
					param.requires_grad = True

		running_loss = 0.0
		pbar = tqdm(train_reader)
		for i, data in enumerate(pbar):
			inputs, labels = data
			inputs, labels = inputs.float().to(device), labels.long().to(device)
			optimizer.zero_grad()
			outputs = net(inputs)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()
			scheduler.step()

			running_loss += loss.item()
			if i % verb_step == verb_step - 1:
				pbar.set_description('Epoch {} Step {}: train cross entropy loss: {:.4f}'.
									 format(epoch, i + 1, running_loss / verb_step))
				running_loss = 0.0

		# validation
		correct = 0
		total = 0
		truth = []
		pred = []
		with torch.no_grad():
			for data in valid_reader:
				inputs, labels = data
				inputs, labels = inputs.float().to(device), labels.long().to(device)
				outputs = net(inputs)
				_, predicted = torch.max(outputs.data, 1)
				total += labels.size(0)
				correct += (predicted == labels).sum().item()
				truth.extend(labels.cpu().numpy())
				pred.extend(predicted.cpu().numpy())

		p, r, f1 = utils.f1_score(truth, pred, 0)
		print('Epoch {}: valid accuracy: {:.2f}, precision: {:.2f}, recall: {:.2f}, f1: {:.2f}'.format(
			epoch, 100 * correct / total, p, r, f1))
		writer.add_scalar('valid/acc', correct / total, epoch)
		writer.add_scalar('valid/precision', p, epoch)
		writer.add_scalar('valid/recall', r, epoch)
		writer.add_scalar('valid/f1', f1, epoch)
		
		#if load_prev == False:
		if False:
			if epoch % save_epoch == 0:
				save_name = os.path.join(save_dir, 'epoch-{}.pth.tar'.format(epoch))
				torch.save({
					'epoch': epochs,
					'state_dict': net.state_dict(),
					'opt_dict': optimizer.state_dict(),
				}, save_name)
				print('Saved model at {}'.format(save_name))

	print('Finished training')
	save_name = str(curr_class_num)+'.pt'
	torch.save({
		'epoch': epochs,
		'state_dict': net.state_dict(),
		'opt_dict': optimizer.state_dict(),
	}, save_name)
	print('Saved model at {}'.format(save_name))

if __name__ == '__main__':
	main()
