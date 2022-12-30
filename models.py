import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as pimg
import os, cv2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import torch, torchvision
from torch import nn
from torchinfo import summary


class ResNet_Model(nn.Module):
	def __init__(self,
		num_classes
	) -> None:
		super(ResNet_Model, self).__init__()
		self.ResNet_Module = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights)
		# self.ResNet_Module.requires_grad_(False)
		self.ResNet_Module.fc = nn.Identity()
		self.fc = nn.Linear(512, num_classes)

	def forward(self, x):
		x = self.ResNet_Module(x)
		x = nn.Flatten()(x)
		x = nn.Dropout(0.25)(x)
		x = self.fc(x)
		return x


# Model = ResNet_Model(67)
# summary(Model, input_size=(1,3,224,224), col_names=[ "input_size", "output_size", "num_params", "trainable"])
