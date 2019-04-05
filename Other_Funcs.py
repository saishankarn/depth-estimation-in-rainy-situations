"""
	Jai Shri Ram
	Swaayatt Robots: March 5, 2019

	Authors: @Sanjeev Sharma @Sunjeet Jena @Sai Shankar N
"""
import numpy as np
import os
import re
import PIL
from PIL import Image


def sort_images(path_images):
	#This function is to sort the required images

	path_=path_images[0].split('_')

	atoi=[ int(i.split('_')[-1].split('.')[0]) for i in path_images]
	atoi=sorted(atoi)
	path_images=[path_[0]+'_'+str(i) + '.jpg' for i in atoi]
	return (path_images)


def read_images_train(path_images, path_folder, N_ROWS, N_COLS):
	#This function is to read the images, resize them and normalize them

	Image_List=[]  #Empty List to store Read and Normalzied Images

	for i in path_images: 

		image_=Image.open(path_folder+'/'+i)
		image_=image_.resize((N_COLS,N_ROWS))
		image_=np.asarray(image_)
		Image_List.append(image_/255.0)

	return Image_List	