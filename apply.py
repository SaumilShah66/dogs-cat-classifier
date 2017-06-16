from __future__ import division, print_function, absolute_import
import numpy as np
import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
import cv2

#Creating and loading trained model
#---------------------------------------------------------
network = input_data(shape=[None, 50, 50, 1])

network = conv_2d(network, 30, 3, activation='relu')
network = max_pool_2d(network, 2)

network = conv_2d(network, 30, 3, activation='relu')
network = max_pool_2d(network, 2)

network = conv_2d(network, 40, 3, activation='relu')
network = max_pool_2d(network, 2)

network = conv_2d(network, 40, 3, activation='relu')
network = max_pool_2d(network, 2)

network = conv_2d(network, 40, 3, activation='relu')
network = max_pool_2d(network, 2)

network = conv_2d(network, 30, 3, activation='relu')
network = max_pool_2d(network, 2)

network = fully_connected(network, 100, activation='relu')
network = dropout(network, 0.5)

network = fully_connected(network, 50, activation='relu')

network = fully_connected(network, 2, activation='softmax')


network = regression(network)

model = tflearn.DNN(network)
print('Model has been created')
print('Loading the network')
model.load('my_cnn.tflearn')
print('Network loaded')
#-------------------------------------------------------
#Loading image
print('Loading an image')
image_name = 'cats/cat.11737.jpg'
img = cv2.imread(image_name,0)
img = cv2.resize(img,(50,50))

# Making prediction
print('Making prediction')
img = np.reshape(img,(-1,50,50,1))
predict = model.predict(img)
if pre[0][0] >= 0.5:
    print("It's a Dog")
elif pre[0][1] >= 0.5:
    print("It's a Cat")
else:
    print("I can't predict")
