from __future__ import division, print_function, absolute_import
import numpy as np
import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import pickle

# Data loading and preprocessing
train_data = pickle.load(open('main_data.p','r'))
test_data = pickle.load(open('test_data.p','r'))

X, Y  = train_data[:,0:2500],train_data[:,2500:]
X_test, Y_test = test_data[:,0:2500],test_data[:,2500:]
X = X.reshape([-1,50,50,1])
X_test = X_test.reshape([-1,50,50,1])

input_layer = input_data(shape=[None, 50, 50, 1],name="input_layer")

conv_1 = conv_2d(input_layer, 30, 3, activation='relu',name="conv_1")
pool_1 = max_pool_2d(conv_1, 2,name="pool_1")

conv_2 = conv_2d(pool_1, 30, 3, activation='relu',name="conv_2")
pool_2 = max_pool_2d(conv_2, 2,name="pool_2")

conv_3 = conv_2d(pool_2, 40, 3, activation='relu',name="conv_3")
pool_3 = max_pool_2d(conv_3, 2,name="pool_3")

conv_4 = conv_2d(pool_3, 40, 3, activation='relu',name="conv_4")
pool_4 = max_pool_2d(conv_4, 2,name="pool_4")

conv_5 = conv_2d(pool_4, 40, 3, activation='relu',name="conv_5")
pool_5 = max_pool_2d(conv_5, 2,name="pool_5")

conv_6 = conv_2d(pool_5, 30, 3, activation='relu',name="conv_6")
pool_6 = max_pool_2d(conv_6, 2,name="pool_6")

fc_1 = fully_connected(pool_6, 100, activation='relu',name="fc_1")
drop = dropout(fc_1, 0.5,name="drop")

fc_2 = fully_connected(drop, 50, activation='relu', name="fc_2")

output = fully_connected(fc_2, 2, activation='softmax',name="output")


# Train using classifier
network = regression(output, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

model = tflearn.DNN(network, tensorboard_verbose=0)
model.fit(X, Y, n_epoch=5, shuffle=True, validation_set=(X_test, Y_test),
          show_metric=True, batch_size=96, run_id='my_cnn')

model.save('my_cnn.tflearn')



conv_1_ = tflearn.variables.get_layer_variables_by_name('conv_1')
# Get a variable's value, using model `get_weights` method:
print("conv_1 layer weights:")
print(model.get_weights(conv_1_[0]))
# Or using generic tflearn function:
# It is also possible to retrieve a layer weights through its attributes `W`
# and `b` (if available).
# Get variable's value, using model `get_weights` method:
print("conv_2 layer weights:")
print(model.get_weights(conv_2.W))

