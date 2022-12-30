"""
Saving test-images and estimating their quality
"""
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
# print (Class2Index)
Index2Class = {v:k for k,v in Class2Index.items()}

# Image Corruptions and Levels
severity_levels = [-1,0,1,2,3]
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

# Saving images with various corruptions
for level in severity_levels:
	if level == -1:
		save_dir = "test_images/severity=" + str(level)
		try:
			os.mkdir(save_dir)
		except:
			None
		main_dir = save_dir
		Save_Images(Data.get_dataset("valid", True), save_dir, Index2Class)
	elif level == 0:
		save_dir = "test_images/severity=" + str(level)
		try:
			os.mkdir(save_dir)
		except:
			None
		Save_Images(Data.get_dataset("valid", True), save_dir, Index2Class)
	else:
		save_dir = "test_images/severity=" + str(level)
		try:
			os.mkdir(save_dir)
		except:
			None
		for corruption_name in corruption_tuple:
			generate_corrupted_images(main_dir, save_dir, corruption_name,level)


Data = []
# Saving images with various corruptions
for level in severity_levels:
	if level == -1:
		dir = "test_images/severity=" + str(level)
		for f in os.listdir(dir):
			img = pimg.imread(dir + "/" + f)

			score, category = predict_image_quality(img, Quality_Estimator)
			true_label = f[:-4]
			pred_label = "No Prediction"
			p = 1
			Data.append([dir + "/" + f, score, category, true_label, pred_label, p])
	elif level == 0:
		dir = "test_images/severity=" + str(level)
		for f in os.listdir(dir):
			img = pimg.imread(dir + "/" + f)

			score, category = predict_image_quality(img, Quality_Estimator)
			true_label = f[:-4]
			pred_label, p = predict_label(img, Classifier, Index2Class)
			Data.append([dir + "/" + f, score, category, true_label, pred_label, p])
	else:
		dir = "test_images/severity=" + str(level)
		for corruption_name in corruption_tuple:
			dir1 = dir + "/" + corruption_name
			for f in os.listdir(dir1):
				img = pimg.imread(dir1 + "/" + f)

				score, category = predict_image_quality(img, Quality_Estimator)
				true_label = f[:-4]
				pred_labe, p = predict_label(img, Classifier, Index2Class)
				Data.append([dir1 + "/" + f, score, category, true_label, pred_label, p])

df = pd.DataFrame(np.array(Data), columns=['image_path', 'global_score', 'category', "true_label", "pred_label", "probability"])
df.to_csv("test_images/mos.csv", index=False)