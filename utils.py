import numpy as np
import matplotlib.pyplot as plt
from random import randint
import cv2
import pickle

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


def resize_image(image, size= (32,32)):

	'''
	resizes the image to a new size and retures a 1D matrix
	size = tuple() = (32,32) default
	'''
	new_image = cv2.resize(image, size).ravel()
	return new_image


def change_color_space(image, cspace):
	if cspace == "YUV":
		return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
	elif cspace == "HLS":
		return cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
	elif cspace == "HSV":
		return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
	elif cspace == "LUV":
		return cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
	elif cspace == "YCrCb":
		return cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)	
	else:
		return np.copy(image)
	

def feature_set_viz(feature_set,title, columns = 4, rows = 10, figsize = (15,15)):
	fig, ax_array = plt.subplots(nrows= rows, ncols= columns, figsize=figsize)
	fig.suptitle(title)
	plt.tight_layout()

	plt.subplots_adjust(top=0.9)
	for i,ax_row in enumerate(ax_array):
		for j,axes in enumerate(ax_row):
			axes.set_yticklabels([])
			axes.set_xticklabels([])
			try:
				axes.set_title(f"channel:[{j-1}]\n{feature_set[i][4]}")
				img = feature_set[i][j]
				img = np.reshape(img, (64,64))
				axes.imshow(img, cmap='gray')
			except:
				axes.set_title('')
				axes.imshow(feature_set[i][j])
	plt.show()


def save_as_pickle(dic, file_path):
	pickle.dump(dic, open(file_path, 'wb') )