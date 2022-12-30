# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import argparse, random
from PIL import Image

import torch, torchvision
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import TQDMProgressBar
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.strategies.ddp import DDPStrategy

from datasets import *
from imagecorruptions import *
from models import *
from paq2piq.paq2piq_standalone import *
from utils import *
from saliency_maps import *

# Dataset
path = "datasets/mit-indoor-scenes"
Data = MIT_Indoor_Scenes_Module(path=path, batch_size=16, num_workers=4)
Class2Index = Data.get_class2index()
Index2Class = {v:k for k,v in Class2Index.items()}


# Image Corruptions and Levels
severity_levels = [0,1,2,3]
corruption_tuple = ("brightness", "contrast", "jpeg_compression", "saturate", "defocus_blur", "motion_blur", "zoom_blur")


# Inference Model of PaQ-2-PiQ
Quality_Estimator = InferenceModel(RoIPoolModel(), 'models/RoIPoolModel.pth')
norm_params = Quality_Estimator.adapt_from_dir('datasets/CLIVE/Images')
Quality_Estimator.normalize(norm_params)

# ResNet Model
Classifier = ResNet_Model(67)
ckpt_path = "checkpoints/mit-indoor-scenes/resnet18/best_model.ckpt"
Classifier = Load_Model(Classifier, ckpt_path)
Classifier.eval()

def Plot_Accuracy_Quality(
	corruption_name,
	Quality_Estimator,
	Classifier,
	Save_Path
):
	Acc1 = []
	Acc5 = []
	Mean_Score = []

	for level in severity_levels: 
		if level != 0:
			Corruption_Function = Corruption(corruption_name, level)
		else:
			Corruption_Function = None
		D = MIT_Indoor_Scenes_Module(path=path, batch_size=16, num_workers=4, corruption=Corruption_Function)
		Dataset = D.get_dataset()
		DataLoader = D.val_dataloader()

		mean_score = predict_quality_dataset(Quality_Estimator, Dataset)
		acc1, acc5 = predict_accuracy_dataloader(Classifier, DataLoader)
		Acc1.append(acc1)
		Acc5.append(acc5)
		Mean_Score.append(mean_score)

	plt.figure(figsize=(18,6))
	plt.subplot(1,3,1)
	plt.plot(np.arange(0,len(severity_levels)), Acc1)
	plt.grid()
	plt.title("Top-1 Accuracy")

	plt.subplot(1,3,2)
	plt.plot(np.arange(0,len(severity_levels)), Acc5)
	plt.grid()
	plt.title("Top-5 Accuracy")

	plt.subplot(1,3,3)
	plt.plot(np.arange(0,len(severity_levels)), Mean_Score)
	plt.grid()
	plt.title("Mean MOS of the dataset")

	plt.savefig(Save_Path + "/" + corruption_name + ".png")

for corruption in corruption_tuple:
	Plot_Accuracy_Quality(corruption, Quality_Estimator, Classifier, "plots")