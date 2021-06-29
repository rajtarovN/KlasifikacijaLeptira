import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import skidanjeSlika
import pickle
from tqdm import tqdm
import random

DATADIR = "C:/Users/rajta/PycharmProjects/oriPrepoznavanjeLeptira/slike"
CATEGORIES=["Pieridae","Nymphalidae","Hesperioidea","Lycaenidae","Papilionidae","Riodinidae"]
 #Ovo je fajl sa ucitavanjem slika, tj povezivanjem slika sa labelama


IMG_SIZE = 64#64 #100 je ok, ili ici na vise


"""
plt.imshow(new_array, cmap='gray')
plt.show()
"""
training_data=[]
all_pictures={}

def revert_dictionary():
    new_dictionary={}
    dictionary_of_ids = skidanjeSlika.ucitaj_csv()
    print(type(dictionary_of_ids))
    for key in dictionary_of_ids.keys():
        for title in dictionary_of_ids[key]:
            new_dictionary[title]=key
            #print("t",title)
            #print("k",key)
    return new_dictionary


def get_ids_pictures():
    err_num=0
    dictionary_of_ids=revert_dictionary()
    create_training_data()
    print(type(dictionary_of_ids))
    for titles in dictionary_of_ids.keys():
        try:
            training_data.append([all_pictures[titles], dictionary_of_ids[titles]])  #sada su slike (jpg spojene sa labelama)
        except KeyError as ke:
            err_num+=1
            print("err",titles)
    print("Ukupno gresaka :",err_num)
    X = []
    y = []
    random.shuffle(training_data)
    for features, label in training_data:
        X.append(features)
        y.append(label)

    #print(X[0].reshape(-1, IMG_SIZE, IMG_SIZE, 3))

    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    pickle_out = open("X.pickle", "wb") #cuvanje podataka
    pickle.dump(X, pickle_out)
    pickle_out.close()

    pickle_out = open("y.pickle", "wb")
    pickle.dump(y, pickle_out)
    pickle_out.close()

    pickle_in = open("X.pickle", "rb")
    print( pickle.load(pickle_in))



def create_training_data():
    path = os.path.join(DATADIR, "")
    for img in tqdm(os.listdir(path)):
        try:
            img_array = cv2.imread(os.path.join(path, img),cv2.IMREAD_GRAYSCALE)  # probacu sa bojama
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

            all_pictures[img.split(".jpg")[0]]=new_array
            #print("i",img.split(".jpg")[0])
            #print("n",new_array)
        except Exception as e:
            print("cc")

        #plt.imshow(img_array)
        #plt.show()
        #break

if __name__=="__main__":
    get_ids_pictures()