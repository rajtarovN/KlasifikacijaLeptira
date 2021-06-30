from datetime import time

import keras
import tensorflow as tf
from keras import regularizers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import numpy
import time
import pickle

from tensorflow.python.keras.utils.np_utils import to_categorical


def cnn():
    """ dense_layers = [0, 1, 2]
    layer_sizes = [32, 64, 128]
    conv_layers = [1, 2, 3]

    X = pickle.load(open("../oriPrepoznavanjeLeptira/X.pickle", "rb"))
    y = pickle.load(open("../oriPrepoznavanjeLeptira/y.pickle", "rb"))
    # print(len(y)) #1972

    X = X / 255.0



     for dense_layer in dense_layers:
         for layer_size in layer_sizes:
             for conv_layer in conv_layers:
     NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
     print(NAME)

     #NAME_LOG="Leptiri-cnn-64x2-{}".format(int(time.time()))


     model=Sequential()
     #----------------prvi lejer
     model.add(Conv2D(layer_size,(3,3),input_shape = X.shape[1:])) #input shape => shape of the data
     model.add(Activation("relu"))
             model.add(MaxPooling2D(pool_size=(2,2)))
             #---------------------- drugi
             for l in range(conv_layer-1):
                 model.add(Conv2D(layer_size,(3,3)))
                 model.add(Activation("relu"))
                 model.add(MaxPooling2D(pool_size=(2,2)))

             #--- 3 lejera
             model.add(Flatten())
             for _ in range(dense_layer):
                 model.add(Dense(layer_size)) #64 noda
                 model.add(Activation("relu"))


             #output
             model.add(Dense(1))
             model.add(Activation('sigmoid'))
             tensor_board = TensorBoard(log_dir='logs/{}'.format(NAME))

             model.compile(loss="categorical_crossentropy",  optimizer="adam",metrics=['accuracy'])

             y=make_list_number(y)
             yi = numpy.array(y)
             #print(y)
             model.fit(X,yi, batch_size=128, epochs=10,validation_split=0.1, callbacks=[tensor_board]) #koliko od jednom valjda radi )#,
             model.save("prvi.model")
     """
    NAME_LOG = "Leptiri-cnn-acc-{}".format(int(time.time()))
    pickle_in = open("../oriPrepoznavanjeLeptira/X.pickle", "rb")
    X = pickle.load(pickle_in)

    pickle_in = open("../oriPrepoznavanjeLeptira/y.pickle", "rb")
    y = pickle.load(pickle_in)

    X = X / 255.0

    model = Sequential()

    model.add(Conv2D(64, (3, 3), input_shape=(64, 64, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(6, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Flatten())

    model.add(Dense(64))
    model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))


    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dense(32))



    model.add(Dense(6, kernel_regularizer=regularizers.l1(0.0001)))
    model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    tensor_board = TensorBoard(log_dir='logs/{}'.format(NAME_LOG))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    y = make_list_number(y)
    yi = numpy.array(y)
    # print(y)
    model.fit(X, yi, batch_size=32, epochs=32, validation_split=0.1, callbacks=[tensor_board])

"""model.add(Dense(64, kernel_regularizer=regularizers.l2(0.001)))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))

    model.add(Dense(3, kernel_regularizer=regularizers.l1(0.001)))
    model.add(Activation("softmax"))"""


#"Pieridae","Nymphalidae","Hesperioidea","Lycaenidae","Papilionidae","Riodinidae"
def make_list_number(y):
 new_y=[]
 #print(y)
 #y = to_categorical(y, 6)

 for i in y:
     if(i==" Pieridae"):
         new_y.append(1)
     elif (i==" Nymphalidae"):
         new_y.append(2)
     elif (i==" Hesperioidea"):
         new_y.append(3)
     elif (i==" Lycaenidae"):
         new_y.append(4)
     elif (i==" Papilionoidea"):
         new_y.append(5)
     else:
         new_y.append(6)

 #new_y = to_categorical(new_y, 7)
 return new_y

if __name__=='__main__':
 cnn()