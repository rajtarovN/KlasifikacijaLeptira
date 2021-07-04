from datetime import time

import keras
import tensorflow as tf
from keras import regularizers
from sklearn.preprocessing import LabelEncoder
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

    NAME_LOG = "Leptiri-cnn-29-6-novo-{}".format(int(time.time()))
    pickle_in = open("../oriPrepoznavanjeLeptira/X.pickle", "rb")
    X = pickle.load(pickle_in)

    pickle_in = open("../oriPrepoznavanjeLeptira/y.pickle", "rb")
    y = pickle.load(pickle_in)

    X = X / 255.0

    model = Sequential()

    model.add(Conv2D(128, (3, 3), input_shape=(64,64,1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

    model.add(Dense(64))#, kernel_regularizer=regularizers.l2(0.0001)))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))

    model.add(Dense(32))  # , kernel_regularizer=regularizers.l2(0.0001)))
    model.add(Activation("relu"))
    model.add(Dropout(0.2))

    model.add(Dense(6, kernel_regularizer=regularizers.l1(0.0001)))
    #model.add(Activation("relu"))


    #model.add(Dense(1))
    model.add(Activation('sigmoid'))
    tensor_board = TensorBoard(log_dir='logs/{}'.format(NAME_LOG))
    opt = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    y = make_list_number(y)
    yi = numpy.array(y)
    #print(yi[0])
    #print(y)
    model.fit(X, yi, batch_size=32, epochs=32,  validation_split=0.1, callbacks=[tensor_board])
    model.save('subota2.model')

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
         new_y.append(0)

     elif (i==" Nymphalidae"):
         new_y.append(1)
     elif (i==" Hesperioidea" or i==" Hesperiidae"):
         new_y.append(2)

     elif (i==" Lycaenidae"):
         new_y.append(3)
     elif (i==" Papilionoidea" or i==" Papilionidae"):
         new_y.append(4)

     else:

         new_y.append(5)




 #new_y =to_categorical(new_y, dtype ="int32") #to_categorical(new_y, 7)
 return new_y

if __name__=='__main__':
 cnn()