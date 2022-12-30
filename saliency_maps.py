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

from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from models import *
from utils import *
from datasets import *
from paq2piq.common import blend_output


def create_saliency_map(
	image_path,
	Model,
	save_path
):
	"""
	Create a Saliency Map for a model.
	Args:
		image_path (string): Path to image.
		Model (torch.nn.Module): Torch model.
		save_path (string): Path to save saliency map.
	"""
	
	Input_Transforms = transforms.Compose([
		transforms.ToTensor(),
		NonRGB(),
		transforms.Resize((224,224)),		
	])

	cam = GradCAM(model=Model, target_layers=[Model.ResNet_Module.layer4[-1]])
	pil_img = Image.open(image_path)
	img = torch.unsqueeze(Input_Transforms(pil_img), dim=0)
	scores = cam(img)
	result = show_cam_on_image(np.array(pil_img)/255.0, scores[0], use_rgb=True)

	blended = Image.fromarray(np.uint8(result*255.0))
	blended.save(save_path)
	

def save_patch_quality_map(
	image_path,
	Model,
	save_path
):
	"""
	Create a Saliency Map for a model.
	Args:
		image_path (string): Path to image.
		Model (torch.nn.Module): Torch model.
		save_path (string): Path to save saliency map.
	"""
	
	img = Image.open(image_path)
	scores = Model.predict_from_pil_image(img)
	blended = blend_output(img, scores, vmin=None, vmax=None)

	blended.save(save_path)