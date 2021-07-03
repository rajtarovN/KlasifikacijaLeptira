import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard

import keras
import numpy
import time
import pickle

def cnn():
    NAME_LOG = "binarno-{}".format(int(time.time()))
    pickle_in = open("../oriPrepoznavanjeLeptira/X.pickle", "rb")
    X = pickle.load(pickle_in)

    pickle_in = open("../oriPrepoznavanjeLeptira/y.pickle", "rb")
    y = pickle.load(pickle_in)

    X = X/255.0

    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=X.shape[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

    model.add(Dense(64))
    model.add(Activation('relu'))

    model.add(Dense(32))
    model.add(Activation('relu'))

    model.add(Dense(1))
    model.add(Activation('softmax'))
    opt = keras.optimizers.Adam(learning_rate=0.001)
    tensor_board = TensorBoard(log_dir='logs/{}'.format(NAME_LOG))
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    y = make_list_number(y)
    yi = numpy.array(y)
    model.fit(X, yi, batch_size=32, epochs=32, validation_split=0.1, callbacks=[tensor_board])


def make_list_number(y):
 new_y=[]
 #print(y)
 #y = to_categorical(y, 6)

 for i in y:
     if(i==" Pieridae"):
         new_y.append(1)
         print("f")
     elif (i==" Nymphalidae"):
         new_y.append(2)
     elif (i==" Hesperioidea" or i==" Hesperiidae"):
         new_y.append(3)
         print("r")
     elif (i==" Lycaenidae"):
         new_y.append(4)
     elif (i==" Papilionoidea" or i==" Papilionidae"):
         new_y.append(5)
         print("e")
     else:
         print(i)
         new_y.append(6)

 #new_y =to_categorical(new_y, dtype ="int32") #to_categorical(new_y, 7)
 return new_y

if __name__=='__main__':
 cnn()