# dogs-cat-classifier

Require dependencies are,
* numpy
* scipy
* opencv-python
* tensorflow
* tflearn


You can get the data from this link.You need to unzip it after downloading it.

http://files.fast.ai/data/dogscats.zip

Your unzipped data set and this repository should be in the same working derectory.
```
~$ ls
~$ dogscats dogscats.zip dogs-cat-classifier
```

This dataset has raw data. Every image has different resolution. We need to read all the images make them of the same resolution. Lables needs to be given as well. We will give [0,1] for cats and [1,0] for dogs. All the images are resized for the resolution of 
50 x 50 and all the images are in grayscale form.
Now change your derectory and run the data.py script. It will create two pickled object file main_data.p and test_data.p.
```
~$ cd dogs-cat-classifier
~/dogs-cat-classifier$ python data.py
```

Now its time to train our model. 
```
~/dogs-cat-classifier$ python model.py
```
After training the model it will create three files 

my_cnn.tflearn.data   

my_cnn.tflearn.meta   

my_cnn.tflearn.index

All of these files are required to use this trained model on any other machine. Keep in mind that it should be used with the same versions of library that are used for training, otherwise it might won't work. You will find all the files in this repository. If you want to use this files than use them with tensorflow 1.0.0 and tflearn 0.3.1. To install this particular versions you can use,
```
~$ sudo pip install -I tensorflow==1.0
~$ sudo pip install -I tflearn==0.3.1
```
Now you can use apply.py script to use your model for new data.
