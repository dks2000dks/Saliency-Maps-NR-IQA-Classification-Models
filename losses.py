import torch,timm
import torch.nn as nn
import torch.nn.functional as F
from timm.loss import SoftTargetCrossEntropy

# Loss Function
def loss_function(
	args
):
	"""
	Getting a loss function.
	"""
	if args.loss == "CrossEntropy":
		# print ("Loss: CrossEntropy")
		return nn.CrossEntropyLoss()

	elif args.loss == "SoftTargetCrossEntropy":
		# print ("Loss: SoftTargetCrossEntropy")
		return SoftTargetCrossEntropy()