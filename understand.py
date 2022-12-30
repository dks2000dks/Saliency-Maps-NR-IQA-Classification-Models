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

# Arguments and Parameters
path = "datasets/mit-indoor-scenes"
Data = MIT_Indoor_Scenes_Module(path=path, batch_size=16, num_workers=4)
Class2Index = Data.get_class2index()
Index2Class = {v:k for k,v in Class2Index.items()}

# Image Corruptions and Levels
severity_levels = [0,1,2,3]
corruption_tuple = ("brightness", "contrast", "jpeg_compression", "saturate", "defocus_blur", "motion_blur", "zoom_blur")


# Saving Saliency Maps
ResNet = ResNet_Model(67)
ResNet = Load_Model(ResNet, "checkpoints/mit-indoor-scenes/resnet18/best_model.ckpt")
ResNet.eval()
PaQ2PiQ = InferenceModel(RoIPoolModel(), 'models/RoIPoolModel.pth')


for level in severity_levels:
	image_dir = "test_images/severity=" + str(level)
	save_dir1 = "saliency_maps/resnet18/severity=" + str(level)
	save_dir2 = "saliency_maps/paq2piq/severity=" + str(level)

	try:
		os.mkdir(save_dir1)
		os.mkdir(save_dir2)
	except:
		None

	if level==0:
		for f in os.listdir(image_dir):
			image_path = image_dir + "/" + f

			# ResNet Saliency Maps
			save_path = save_dir1 + "/" + f
			create_saliency_map(image_path, ResNet, save_path)

			# PaQ-2-PiQ
			save_path = save_dir2 + "/" + f
			save_patch_quality_map(image_path, PaQ2PiQ, save_path)
	else:
		for corrruption_name in corruption_tuple:
			for f in os.listdir(image_dir + "/" + corrruption_name):
				image_path = image_dir + "/" + corrruption_name + "/" + f

				# ResNet Saliency Maps
				try:
					os.mkdir(save_dir1 + "/" + corrruption_name)
				except:
					None
				save_path = save_dir1 + "/" + corrruption_name + "/" + f
				create_saliency_map(image_path, ResNet, save_path)

				# PaQ-2-PiQ
				try:
					os.mkdir(save_dir2 + "/" + corrruption_name)
				except:
					None
				save_path = save_dir2 + "/" + corrruption_name + "/" + f
				save_patch_quality_map(image_path, PaQ2PiQ, save_path)
