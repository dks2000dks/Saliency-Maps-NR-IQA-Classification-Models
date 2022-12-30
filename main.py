# Importing Libraries
import os

import torch, timm
from torch import nn
import torchmetrics
from torchinfo import summary
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.strategies.ddp import DDPStrategy

import datasets
import models
import utils
import arguments
import losses
import optimizers_schedulers 


# Lightning Module
class Model_LightningModule(pl.LightningModule):
	def __init__(self, args):
		super().__init__()
		self.args = args

		# Model
		self.model = Training_Model
		self.save_hyperparameters()

		# Loss
		self.train_lossfn = losses.loss_function(args)
		self.val_lossfn = nn.CrossEntropyLoss()

		# Metrics
		self.acc_top1 = torchmetrics.Accuracy()
		self.acc_top5 = torchmetrics.Accuracy(top_k=5)
		self.valid_acc_top1 = torchmetrics.Accuracy()
		self.valid_acc_top5 = torchmetrics.Accuracy(top_k=5)

	# Training-Step
	def training_step(self, batch, batch_idx):
		original_x, original_y = batch
		logits = self.model(original_x)

		train_loss = self.train_lossfn(logits, original_y)
		self.acc_top1(logits, original_y)
		self.acc_top5(logits, original_y)

		self.log('acc_top1', self.acc_top1, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
		self.log('acc_top5', self.acc_top5, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)

		return train_loss

	# Validation-Step
	def validation_step(self, batch, batch_idx):
		x, y = batch
		logits = self.model(x)
		
		val_loss = self.val_lossfn(logits, y)
		self.valid_acc_top1(logits, y)
		self.valid_acc_top5(logits, y)

		self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
		self.log('val_acc_top1', self.valid_acc_top1, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
		self.log('val_acc_top5', self.valid_acc_top5, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

	# Configure Optimizers
	def configure_optimizers(self):
		optimizer = optimizers_schedulers.optimizers(parameters=self.model.parameters(), args=self.args)
		scheduler = optimizers_schedulers.schedulers(optimizer=optimizer, args=self.args)
		if scheduler is None:
			return optimizer
		else:
			return [optimizer], [scheduler]


# Main Function
def main(args):
	# Names
	model_name = args.model_name
	dataset_name = args.dataset_name


	# Get Datasets
	Train_Dataloader = Dataset.train_dataloader()
	Valid_Dataloader = Dataset.val_dataloader()


	# Lightning Module
	Model = Model_LightningModule(args)


	# Checkpoint Callbacks
	best_checkpoint_callback = ModelCheckpoint(
		save_top_k=1,
		monitor="val_loss",
		mode="min",
		dirpath=args.main_path + "checkpoints/" + dataset_name + "/" + model_name,
		filename="best_model",
	)
	last_checkpoint_callback = ModelCheckpoint(
		dirpath=args.main_path + "checkpoints/" + dataset_name + "/" + model_name,
		save_last=True
	)


	# Training
	if args.ckpt_path is not None:
		if os.path.isfile(args.ckpt_path):
			print ("Found the checkpoint.")
		else:
			args.ckpt_path = None
			print ("No Checkpoint found in the path. Starting training from the begining.")
	else:
		print ("Starting training from the begining.")


	# PyTorch Lightning Trainer
	trainer = pl.Trainer(
		accelerator="gpu",
		strategy=DDPStrategy(find_unused_parameters=False),
		devices = args.gpu,
		callbacks=[best_checkpoint_callback, last_checkpoint_callback, utils.LitProgressBar()],
		num_nodes=args.num_nodes,
		max_epochs=args.epochs,
		logger=pl_loggers.TensorBoardLogger(save_dir=args.main_path)
	)


	# Training the Model
	if args.train:
		print ("-"*25 + " Starting Training " + "-"*25)
		trainer.fit(Model, train_dataloaders=Train_Dataloader, val_dataloaders=Valid_Dataloader, ckpt_path=args.ckpt_path)
		trainer.validate(Model, Train_Dataloader, ckpt_path=args.ckpt_path)
		trainer.validate(Model, Valid_Dataloader, ckpt_path=args.ckpt_path)


	# Evaluate the Model
	if args.evaluate:
		print ("-"*25 + " Starting Evaluation " + "-"*25)
		trainer.validate(Model, Valid_Dataloader, ckpt_path=args.ckpt_path)


# Calling Main function
if __name__ == '__main__':
	root_dir = os.path.dirname(os.path.realpath(__file__))

	# Get Arguments
	args = arguments.Parse_Arguments()

	# No.of Classes in Dataset
	Num_Classes = 67

	# Training Model
	Training_Model = models.ResNet_Model(Num_Classes)
	# summary(Training_Model, input_shape=(1,3,224,224), col_names=("input_size","output_size","num_params","mult_adds"), col_width=17)

	# Dataset
	Dataset = datasets.MIT_Indoor_Scenes_Module(args.data, args.batch_size, corruption=None)

	# Set-Arguments
	args.model_name = "resnet18"
	args.dataset_name = "mit-indoor-scenes"

	# Main Function
	main(args)