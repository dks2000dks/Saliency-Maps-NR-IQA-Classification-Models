import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as pimg
import os,pandas as pd
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import ImageFolder, CIFAR10
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from typing import Callable, Optional, Any

from imagecorruptions import *


class Corruption(object):
	"""
	Stack a single channel 3 times for Non-RGB Images
	"""
	def __init__(self, corruption_name, severity) -> None:
		self.corruption_name = corruption_name
		self.severity = severity

	def __call__(self, img):
		if self.corruption_name == "resize":
			return transforms.Resize((224,224))(img)
		else:
			img_numpy = img.detach().numpy().transpose(1,2,0)
			corrupt_img = corrupt(np.uint8(img_numpy*255.0), corruption_name=self.corruption_name, severity=self.severity).transpose(2,0,1)/255.0
			return torch.as_tensor(corrupt_img, dtype=img.dtype)


class NonRGB(object):
	"""
	Stack a single channel 3 times for Non-RGB Images
	"""
	def __call__(self, img):
		if img.shape[0] == 1:
			return torch.concat((img,img,img), dim=0)
		else:
			return img


class ImageFolder_Masking(ImageFolder):
	"""
	Extends torchvision.datasets.ImageFolder. It is a custom dataset that includes image file paths.
	Source: https://gist.github.com/andrewjong/6b02ff237533b3b2c554701fb53d5c4d
	"""
	# Overriding
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

		# Using MOS.csv for quality ratings
		df = pd.read_csv(self.root + "/mos.csv")
		Data = df.to_numpy()
		self.ImagePath2QualityCategory = dict(zip(Data[:,0], Data[:,3]))

		updated_imgs = self.imgs
		self.imgs = []
		for i in range(len(updated_imgs)):
			if self.ImagePath2QualityCategory[updated_imgs[i][0]] == "Excellent" or self.ImagePath2QualityCategory[updated_imgs[i][0]] == "Good":
				self.imgs.append(updated_imgs[i])


class MIT_Indoor_Scenes_Module(pl.LightningDataModule):
	def __init__(self, 
		path,
		batch_size,
		corruption = None,
		num_workers = 16,
	) -> None:
		"""
		MIT Indoor Scenes dataset Module
		Args:
			path (string): The path to MIT Indoor Scenes dataset folder.
			batch_size (int): The batch-size for dataloaders.
			num_workers (int): No.of to process dataset.
			num_classes (int): No.of calasses in MIT Indoor Scenes dataset.
		"""
		super().__init__()
		self.path = path
		self.batch_size = batch_size
		self.num_workers = num_workers
		self.targets = len(os.listdir(self.path))
		self.corruption = corruption

		self.original_dataset = ImageFolder_Masking(self.path, transform=transforms.Compose([transforms.ToTensor(),NonRGB()]))
		TrainIndices, ValidIndices = train_test_split(np.arange(len(self.original_dataset.targets)),test_size=0.2,stratify=self.original_dataset.targets, random_state=32)
		self.original_train = torch.utils.data.Subset(self.original_dataset, TrainIndices)
		self.original_valid = torch.utils.data.Subset(self.original_dataset, ValidIndices)

		self.dataset = ImageFolder_Masking(self.path, transform=self.transform())
		self.train = torch.utils.data.Subset(self.dataset, TrainIndices)
		self.valid = torch.utils.data.Subset(self.dataset, ValidIndices)
	
		
	def transform(self):
		"""
		Input Image Transformation
		"""
		if self.corruption is None:
			Input_Transforms = transforms.Compose([
				transforms.ToTensor(),
				NonRGB(),
				transforms.CenterCrop((224,224))
			])
			return Input_Transforms
		else:
			Input_Transforms = transforms.Compose([
				transforms.ToTensor(),
				NonRGB(),
				self.corruption,
				transforms.CenterCrop((224,224))
			])
			return Input_Transforms
	
	def train_dataloader(self):
		return DataLoader(self.train, batch_size=self.batch_size, num_workers = self.num_workers, shuffle=True)

	def val_dataloader(self):
		return DataLoader(self.valid, batch_size=self.batch_size, num_workers = self.num_workers)

	def test_dataloader(self):
		return DataLoader(self.valid, batch_size=self.batch_size, num_workers = self.num_workers)

	def get_dataset(self,
		split='valid',
		resize=True
	):
		"""
		Getting dataloader with and without any image resizing
		Args:
			split (string): Split of dataset. Options: ["train", "valid"]
			resize (boolean): Whether to resize images or not
		"""
		if split=='train':
			if resize:
				return self.train
			else:
				return self.original_train
		if split=='valid':
			if resize:
				return self.valid
			else:
				return self.original_valid
	
	def get_class2index(self):
		return self.dataset.class_to_idx