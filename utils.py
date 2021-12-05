import numpy as np
import matplotlib.pyplot as plt
from random import randint

def get_img_size(dir):
	'''returns the dimensions of images'''
	for i in range(2):
		img = plt.imread(dir[randint(1,len(dir))]) 
		print(f"The size of image is {img.shape[0]} by {img.shape[1]}")


def show_random_images(image_dir,rows=2, cols=10, num=20, title = ""):

	'''
	args:
	rows, cols: numbe of rows and columns for the subplots
	num: number of images
	image_dir: list of paths to image
	title: title of the plot
	
	output: random images
	'''

	figure, ax = plt.subplots(nrows=rows,ncols=cols, figsize = (15,6))
	
	for i in range(num):
		n = randint(1,5000)
		img = plt.imread(image_dir[n])
		ax.ravel()[i].imshow(img, cmap = "gray")
		plt.tight_layout()
		plt.suptitle(title, size = 15)
		plt.subplots_adjust(top=1.3)
	plt.show()