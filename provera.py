import cv2
import tensorflow as tf

CATEGORIES=["Pieridae","Nymphalidae","Hesperioidea","Lycaenidae","Papilionidae","Riodinidae"]


def prepare(filepath):
    IMG_SIZE = 64  # 50 in txt-based
    img_array = cv2.imread(filepath)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)




model = tf.keras.models.load_model("64x3-CNN.model")

prediction = model.predict([prepare("C:\\Users\\rajta\\PycharmProjects\\oriPrepoznavanjeLeptira\\validation\\012824396_Zeuxidia_amethystus_davidi_Monastyrskii_2013_PT_1.jpg")])
print(prediction)  # will be a list in a list.
#print(CATEGORIES[int(prediction[0][0])])