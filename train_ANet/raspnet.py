import torch
from torch import nn

class ONet(nn.Module):

	def __init__(self, class_num):
		super().__init__()

		self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
		self.prelu1 = nn.PReLU(32)
		self.pool1 = nn.MaxPool2d(3, 2, ceil_mode=True)
		self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
		self.prelu2 = nn.PReLU(64)
		self.pool2 = nn.MaxPool2d(3, 2, ceil_mode=True)
		self.conv3 = nn.Conv2d(64, 64, kernel_size=3)
		self.prelu3 = nn.PReLU(64)
		self.pool3 = nn.MaxPool2d(2, 2, ceil_mode=True)
		self.conv4 = nn.Conv2d(64, 128, kernel_size=2)
		self.prelu4 = nn.PReLU(128)
		self.dense5 = nn.Linear(1152, 256)
		self.prelu5 = nn.PReLU(256)
		self.dense6_1 = nn.Linear(256, class_num) # age

		self.training = True

	def forward(self, x):
		x = self.conv1(x)
		x = self.prelu1(x)
		x = self.pool1(x)
		x = self.conv2(x)
		x = self.prelu2(x)
		x = self.pool2(x)
		x = self.conv3(x)
		x = self.prelu3(x)
		x = self.pool3(x)
		x = self.conv4(x)
		x = self.prelu4(x)
		x = self.dense5(x.view(x.shape[0], -1))
		x = self.prelu5(x)
		a = self.dense6_1(x)
		return a

def raspnet(**kwargs):
	if kwargs['name'] == 'onet_a':
		model = ONet(kwargs['class_num'])
	else:
		raise NotImplementedError
	return model

if __name__ == '__main__':
	pass

