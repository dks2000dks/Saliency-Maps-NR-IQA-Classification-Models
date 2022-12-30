import argparse

def Parse_Arguments():
	# Argument Parser
	parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')


	# Dataset Paths
	parser.add_argument('data', default='imagenet', help='Path to dataset. (default: imagenet)')
	parser.add_argument('--num-workers', default=64, type=int, help='No.of data loading workers. (default: 64)')
	parser.add_argument('--image-shape', default=(224,224), type=tuple, help='Dimensions of image during training. (default: (224,224))')


	# Mode
	parser.add_argument('--train', action='store_true', help='Training model on training dataset and simultaneouly validating on validation dataset.')
	parser.add_argument('--evaluate', action='store_true', help='Evaluating model on validation dataset.')


	# Path
	parser.add_argument('--main-path', default='', type=str, help='Path to main.py.')
	parser.add_argument('--ckpt-path', default=None, type=str, help='Path to checkpoints to resume training.')


	# Training Parameters
	parser.add_argument('--epochs', default=60, type=int, help='No.of total epochs for training. (default: 300)')
	parser.add_argument('--batch-size', default=32, type=int, help='Mini-batch size during training. This is the total batch size of all GPUs on the current node when using  Distributed Data Parallel Strategy. (default: 16)')
	

	# Loss Function
	parser.add_argument('--loss', default="CrossEntropy", type=str, help='Loss Function for training. (options: "CrossEntropy", "SoftTargetCrossEntropy") (default: "CrossEntropy")')
	
	# Optimizer and it's Parameters
	parser.add_argument('--optimizer', default="SGD", type=str, help='Optimizer for training. (options: "Adam", "AdamW", "SGD", "Adagrad") (default: "SGD")')
	parser.add_argument('--lr', default=0.001, type=float, help='Initial learning rate. (default: 0.001)')
	parser.add_argument('--momentum', default=0.0, type=float, help='Momentum for SGD Optimizer. (default: 0.9)')
	parser.add_argument('--weight-decay', default=0.001, type=float, help='Weight Decay for all optimizers. (default: 0.05)')


	# Schedulers and it's Parameters
	parser.add_argument('--scheduler', default="None", type=str, help='Scheduler for learning-rate during training. (options: "CosineAnnealingLR", "CosineAnnealingWarmRestarts") (default: "None")')
	parser.add_argument('--warmup-epochs', default=5, type=int, help='No.of warmup epochs for cosine scheduler. (default: 5)')
	parser.add_argument('--multiplier', default=59, type=int, help='Multiplier of no.of warmup epochs for next cycle. (default: 59)')

	
	# Distributed Training Parameters
	parser.add_argument('--num-nodes', default=1, type=int, help='Number of nodes for distributed training (default: 1).')
	parser.add_argument('--gpu', default=1, type=int, help='Number of GPUs per nodes for distributed training (default: 1).')
	
	
	# Model and Dataset Names
	parser.add_argument('--model-name', default="", type=str, help="Name of model (Used during saving).")
	parser.add_argument('--dataset-name', default="", type=str, help="Name of dataset (Used during saving).")

	return parser.parse_args()
