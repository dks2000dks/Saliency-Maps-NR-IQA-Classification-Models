import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as pimg
import os,fastai,pandas as pd
import warnings
warnings.filterwarnings('ignore')

import torch, torchmetrics
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import ImageFolder, CIFAR10
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.callbacks import TQDMProgressBar
import pytorch_lightning as pl
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from typing import Callable, Optional, Any

from paq2piq.paq2piq_standalone import *
from imagecorruptions import *


# Progress bar
class LitProgressBar(TQDMProgressBar):
	def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
		print ()
		return super().on_validation_epoch_end(trainer, pl_module)


def predict_image_quality_of_dataset(
	path
):
	"""
	Predict quality of a dataset of images.
	Args:
		path: Path to an image.
	"""
	# Categories of Quality Rating
	Categories = ['Bad', 'Poor', 'Fair', 'Good', 'Excellent']

	# Inference Model of PaQ-2-PiQ
	Model = InferenceModel(RoIPoolModel(), 'models/RoIPoolModel.pth')

	# Adapting Model to CLIVE dataset
	norm_params = Model.adapt_from_dir('datasets/CLIVE/Images')
	Model.normalize(norm_params)

	# Data
	Data = []
	for _class in os.listdir(path):
		class_path = path + _class
		if os.path.isdir(class_path):
			for image_file in os.listdir(class_path):
				# Image Path
				image_path = class_path + "/" + image_file

				# Image
				img = Image.open(image_path)
				# img = img.resize((224,224))

				# Predicting MOS Score
				mos_score = Model.predict_from_pil_image(img)

				Data.append([image_path, mos_score["global_score"], mos_score["normalized_global_score"], mos_score["category"]])
		else:
			None

	# Pandas Dataframe
	df = pd.DataFrame(np.array(Data), columns=['image_path', 'global_score', 'normalized_global_score', 'category'])

	df.to_csv(path + "mos.csv", index=False)


def Save_Images(
	Dataset,
	path,
	Index2Class,
	num_images = 1
) -> None:
	"""
	Save some images in dataloaders.
	Args:
		Dataset (torch.data.dataset): PyTorch dataset.
		path (string): Path to save images
		Index2Class (dict): Index to Class mapping.
		nm_images (int): No.of images to save
	"""
	# Generating a random image. "5" is set randomly.
	num_saved_images = 0
	saved_image_labels = []
	i = 0

	while num_saved_images < num_images:
		img, label = Dataset.__getitem__(i+5)
		i += 1

		label = Index2Class[label]

		img = np.uint8(255.0*img.numpy().transpose(1,2,0))

		img = Image.fromarray(img)
		img.save(path + "/" + label + ".png")
		num_saved_images += 1


def generate_corrupted_images(
	images_path,
	save_path,
	corruption_name,
	severity
):
	"""
	Generate corrupted images. Corrupted Images of original dimensions are saved after getting resized to (224,224,3)
	Args:
		images_path (string):  Path to images
		save_path (string): Path where images shoudl be saved
		corruption_name (string): Name of corruption. Corruptions available are
			("gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur",
			"glass_blur", "motion_blur", "zoom_blur", "snow", "frost", "fog",
			"brightness", "contrast", "elastic_transform", "pixelate",
			"jpeg_compression", "speckle_noise", "gaussian_blur", "spatter",
			"saturate")
		severity (int): Levels of severity. Available levels [-1,0,1,2,3,4,5]
	"""
	
	for f in os.listdir(images_path):
		img = np.uint8(255.0 * pimg.imread(images_path + "/" + f))
		corrupt_img = Image.fromarray(corrupt(img, corruption_name=corruption_name, severity=severity))

		try:
			os.mkdir(save_path + "/" + corruption_name)
		except:
			None

		transform = transforms.CenterCrop((224,224))
		corrupt_img = transform(corrupt_img)
		corrupt_img.save(save_path + "/" +corruption_name + "/" + f)


def Load_Model(
	Model: torch.nn.Module,
	Path
):
	"""
	Loading weights to a PyTorch Model
	Args:
		Model (torch.nn.Module): PyTorch Model
		Path (string): path to best checkpoint save by PyTorch Lightning trainer.
	Returns:
		Model (torch.nn.Module): PyTorch Model with trained weights.
	"""
	checkpoint = torch.load(Path)

	# Model Weights
	model_weights = checkpoint["state_dict"]
	for key in list(model_weights):
		model_weights[key.replace("model.", "")] = model_weights.pop(key)

	Model.load_state_dict(model_weights)

	return Model


def predict_accuracy_dataloader(
	Model,
	Dataloader
):
	Acc1 = torchmetrics.Accuracy(top_k=1, num_classes=67)
	Acc5 = torchmetrics.Accuracy(top_k=5, num_classes=67)

	for data in Dataloader:
		images, labels = data
		images = images.cuda()
		Model.cuda()
		labels = labels.cuda()
		outputs = Model(images)
		
		Acc1(outputs.cpu(), labels.cpu())
		Acc5(outputs.cpu(), labels.cpu())

	return float(Acc1.compute().numpy()), float(Acc5.compute().numpy())


def predict_quality_dataset(
	Model,
	dataset
):
	Quality = []
	for i in range(len(dataset)):
		img = dataset.__getitem__(i)[0].numpy()
		score, _ = predict_image_quality(img.transpose(1,2,0), Model)
		Quality.append(score)
	return np.mean(Quality)


def predict_image_quality(
	img,
	Model,
):
	"""
	Predict quality of a ".jpg" image.
	Args:
		img: Numpy array of image with range [0,1] and shape (224,224).
	Returns:
		mos_score: Quality score of image.
	"""

	# Image
	img = Image.fromarray(np.uint8(255.0 * img))

	# Predicting MOS Score
	mos_score = Model.predict_from_pil_image(img)
	return mos_score["global_score"], mos_score['category']


def predict_label(
	img,
	Model,
	Index2Class
):
	"""
	Predict quality of a ".jpg" image.
	Args:
		img: Numpy array of image with range [0,1] and shape (224,224).
	Returns:
		mos_score: Quality score of image.
	"""

	# Image
	img = torch.Tensor(np.expand_dims(img.transpose(2,0,1), axis=0)).cuda()
	Model.cuda()
	pred = Model(img)
	prob = F.softmax(pred)
	_, predicted = torch.max(prob, 1)
	label = Index2Class[int(predicted[0])]
	return label, float(prob[:,int(predicted[0])].cpu().detach().numpy())
