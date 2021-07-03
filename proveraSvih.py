import os

import cv2
import keras
import tensorflow as tf
from tqdm import tqdm

CATEGORIES=["Pieridae","Nymphalidae","Hesperioidea","Lycaenidae","Papilionidae","Riodinidae"]


def prepare(filepath):
    IMG_SIZE = 64  # 50 in txt-based
    #img_array = cv2.imread(filepath)
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)



model = tf.keras.models.load_model("cetvrtak.model")
slike=[]
for img in tqdm(os.listdir("C:/Users/rajta/PycharmProjects/oriPrepoznavanjeLeptira/validation")):
    slike.append(prepare(img))

for slika in slike:
    prediction = model.predict([slika])
    print(prediction)  # will be a list in a list.
#print(CATEGORIES[int(prediction[0][0])])


opt = keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy',
optimizer=opt, #optimizers.RMSprop(lr=1e-4),
metrics=['acc'])

#img = cv2.imread('E:/tulip.jpg')
#img = cv2.resize(img,(150,150))
#img = np.reshape(img,[1,150,150,3])

#classes = model.predict([prepare("C:\\Users\\rajta\\PycharmProjects\\oriPrepoznavanjeLeptira\\validation\\012824377_Udara_renevieri_Braby & Muller_2013_PT.jpg")])
#print(classes)