import os

import cv2
import keras
import tensorflow as tf
from tqdm import tqdm

CATEGORIES=["Pieridae","Nymphalidae","Hesperioidea","Lycaenidae","Papilionidae","Riodinidae"]


def make_dictionary():
    f = open("slikePoklasama.txt", "r")
    all_rows=f.readlines()
    dictionary_of_titles={}
    for row in all_rows:
        dictionary_of_titles[row.split(".jpg ")[0]]=row.split(".jpg ")[1]
        #print(dictionary_of_titles[row.split(".jpg ")[0]], )
    return dictionary_of_titles



def prepare(filepath):
    IMG_SIZE = 64  # 50 in txt-based
    #img_array = cv2.imread(filepath)
    try:
        img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    except:
        pass


def calculate_prediction(prediction_list, nazivi):
    dictionary_labels_img=make_dictionary()
    i=0
    tacnost_testa=0
    greske_testa=0
    for one_prediction in prediction_list:
        j=0

        predvidjena_klasa=False
        for item in one_prediction[0]:

            if item!=0:
                label=""
                label2=""
                if j==0:
                    label="Pieridae"
                elif j==1:
                    label="Nymphalidae"
                elif j==2:
                    label="Hesperioidea"
                    label2="Hesperiidae"
                elif j==3:
                    label="Lycaenidae"
                elif j==4:
                    label="Papilionoidea"
                    label2="Papilionidae"

                elif j == 5:
                    label = "Riodinidae"
                else:
                    label=""
                it_str=str(item)
                #print(nazivi[i]+" have "+it_str+" % of "+label+" actualy it is type: "+ dictionary_labels_img[nazivi[i].split(".jpg")[0]])
                #print((label==dictionary_labels_img[nazivi[i].split(".jpg")[0]] or label2==dictionary_labels_img[nazivi[i].split(".jpg")[0]]))
                if((label==dictionary_labels_img[nazivi[i].split(".jpg")[0]].strip() or label2==dictionary_labels_img[nazivi[i].split(".jpg")[0]].strip())):
                    tacnost_testa+=1
                    predvidjena_klasa=True
                else:
                    if( not predvidjena_klasa):
                        #print(nazivi[i] + " have " + it_str + " % of " + label + " actualy it is type: " +
                        #      dictionary_labels_img[nazivi[i].split(".jpg")[0]])
                        #print(one_prediction)
                        #print(label)
                        #print(label2)
                        #print(dictionary_labels_img[nazivi[i].split(".jpg")[0]])
                        #print("-----------------------------------------------")
                        greske_testa+=1
            j+=1
        i+=1
    print(tacnost_testa)
    print(greske_testa)
    krajnje=(100*tacnost_testa)/i
    print("Accuracy of test: "+ str(krajnje))




def check():
    model = tf.keras.models.load_model("subota.model")
    slike=[]
    nazivi=[]
    for img in tqdm(os.listdir("C:/Users/rajta/PycharmProjects/oriPrepoznavanjeLeptira/validation")):
        slike.append(prepare(os.path.join("C:/Users/rajta/PycharmProjects/oriPrepoznavanjeLeptira/validation", img)))
        nazivi.append(img)


    prediction_list=[]
    for slika in slike:
        try:
            prediction = model.predict([slika])
            prediction_list.append(prediction) #ovde sam sad dobila listu predikcija, a gore imam recnik naziva, i imam listu naziva
        except:
            pass

    calculate_prediction(prediction_list,nazivi)
        #print(prediction)  # will be a list in a list.
    #print(CATEGORIES[int(prediction[0][0])])




    #opt = keras.optimizers.Adam(learning_rate=0.001)
    #model.compile(loss='categorical_crossentropy',
    #optimizer=opt, #optimizers.RMSprop(lr=1e-4),
    #metrics=['acc'])

    #img = cv2.imread('E:/tulip.jpg')
    #img = cv2.resize(img,(150,150))
    #img = np.reshape(img,[1,150,150,3])

    #classes = model.predict([prepare("C:\\Users\\rajta\\PycharmProjects\\oriPrepoznavanjeLeptira\\validation\\012824377_Udara_renevieri_Braby & Muller_2013_PT.jpg")])
    #print(classes)

if __name__=='__main__':
 check()