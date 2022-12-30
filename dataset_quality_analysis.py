# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as pimg
import os,pandas as pd
import warnings
warnings.filterwarnings('ignore')

from utils import *

# Use the line below to estimate quality of the dataset.
predict_image_quality_of_dataset("datasets/mit-indoor-scenes/")

# Use the line below to save some of the validation examples


df = pd.read_csv("datasets/mit-indoor-scenes/mos.csv", sep=",")
print (df.columns.values)																	# Names of columns in the data
print (df.describe())																		# Gives statitics of the data

# NumPy Array
Data = df.to_numpy()

def Plot_Histogram(
	x, 
	title, 
	xlabel, 
	ylabel, 
	f
):
	counts, bins = np.histogram(x, bins=100)

	plt.figure(figsize=(6,6))
	plt.title(title)
	plt.grid()
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.hist(bins[:-1], bins, weights=counts, color="tab:blue")
	plt.savefig("results/" + f)
	plt.show()

def Bar_Plot(
	x, 
	title, 
	xlabel, 
	ylabel, 
	f
):
	Categories, Frequency = np.unique(x, return_counts=True)
	print (Categories, Frequency)
	plt.figure(figsize=(6,6))
	plt.title(title)
	plt.grid()
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.bar(Categories, Frequency)
	plt.savefig("results/" + f)
	plt.show()


Plot_Histogram(Data[:,1], "Histogram of predicted MOS\nof MIT-Indoor-Scenes dataset using PaQ-2-PiQ", "MOS", "Frequency", "MOS_Histogram.png")
Plot_Histogram(Data[:,2], "Histogram of predicted Normalized MOS\nof MIT-Indoor-Scenes \n dataset using PaQ-2-PiQ", "Normalized MOS", "Frequency", "Normalized_MOS_Histogram.png")
Bar_Plot(Data[:,3], "Bar plot of predicted Quality Category\nof MIT-Indoor-Scenes \n dataset using PaQ-2-PiQ", "Quality Category", "Frequency", "Quality_Category_Histogram.png")