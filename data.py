import cv2
import numpy as np
import glob
import sys
import os
import pickle

# For doga [1,0]
# For cats [0,1]
def create_data():
	image_size = 50
	length = image_size*image_size
	os.chdir('dogscats/train/dogs')
	pics_of_dogs = glob.glob('*.jpg')
	os.chdir('../cats')
	pics_of_cats = glob.glob('*.jpg')
	os.chdir('..')
	total_images = len(pics_of_cats) + len(pics_of_dogs)
	main_data = np.zeros([total_images,length+2],dtype='f')
	i=0
	os.chdir('dogs')
	print("Reading images of dogs")
	for pic in pics_of_dogs:
		img = cv2.imread(pic,0)
		img = cv2.resize(img,(image_size,image_size))
		main_data[i,0:length] = np.reshape(img,(1,length))
		main_data[i,length] = 1
		i = i+1
	print("Reading images of cats")
	os.chdir('../cats')
	for pic in pics_of_cats:
		img = cv2.imread(pic,0)
		img = cv2.resize(img,(image_size,image_size))
		main_data[i,0:length] = np.reshape(img,(1,length))
		main_data[i,length+1] = 1
		i = i+1
	os.chdir('../../..')
	np.random.shuffle(main_data)
	print("saving data set")
	pickle.dump(main_data,open('main_data.p','w'))
	return 
	
def test_data():
	image_size = 50
	length = image_size*image_size
	os.chdir('dogscats/valid/dogs')
	pics_of_dogs = glob.glob('*.jpg')
	os.chdir('../cats')
	pics_of_cats = glob.glob('*.jpg')
	os.chdir('..')
	total_images = len(pics_of_cats) + len(pics_of_dogs)
	test_data = np.zeros([total_images,length+2],dtype='f')
	i=0
	os.chdir('dogs')
	print("reading images of dogs")
	for pic in pics_of_dogs:
		img = cv2.imread(pic,0)
		img = cv2.resize(img,(image_size,image_size))
		test_data[i,0:length] = np.reshape(img,(1,length))
		test_data[i,length] = 1
		i = i+1
	os.chdir('../cats')
	print("Reading images of cats")
	for pic in pics_of_cats:
		img = cv2.imread(pic,0)
		img = cv2.resize(img,(image_size,image_size))
		test_data[i,0:length] = np.reshape(img,(1,length))
		test_data[i,length+1] = 1
		i = i+1
	os.chdir('../../..')
	np.random.shuffle(test_data)
	print("Saving testing data set")
	pickle.dump(test_data,open('test_data.p','w'))	
	return
create_data()
test_data()
