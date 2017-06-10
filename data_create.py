import cv2
import numpy as np
import glob
import sys
import os

os.chdir('dogs')
pics_of_dogs = glob.glob('*.jpg')
os.chdir('../cats')
pics_of_cats = glob.glob('*.jpg')
total_images = len(pics_of_cats) + len(pics_of_dogs)
main_data = np.zeros([total_images,2501])
i=0

# for dog 0
# for cat 1
os.chdir('../dogs')
for pic in pics_of_dogs:
	img = cv2.imread(pic,0)
	img = cv2.resize(img,(50,50))
	main_data[i,0:2500] = np.reshape(img,(1,2500))
	main_data[i,2500] = 0
	i = i+1
	print(i)

os.chdir('../cats')
for pic in pics_of_cats:
	img = cv2.imread(pic,0)
	img = cv2.resize(img,(50,50))
	main_data[i,0:2500] = np.reshape(img,(1,2500))
	main_data[i,2500] = 1
	i = i+1
	print(i)
os.chdir('..')

np.random.shuffle(main_data)
file = open('data.csv','w')
for j in range(len(main_data)):
	for k in range(len(main_data[j,:])):
		file.write(str(main_data[j,k])
		file.write(',')
	file.write('\n')
file.close()
