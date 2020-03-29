
import os
import numpy as np
from PIL import Image
import cv2 
import pickle

face_classifier = cv2.CascadeClassifier('resources/classifiers/haarcascade_frontalface_alt2.xml')
fisher_recognizer = cv2.face.FisherFaceRecognizer_create()
lbphf_recognizer = cv2.face.LBPHFaceRecognizer_create()

#Directorios de trabajo
base_dir=os.path.dirname(os.path.abspath(__file__))
img_dir=os.path.join(base_dir, "resources/faces_2_recognize")

current_id=0
label_ids={}#Contiene la asociacion label(str)-label(int)


                
#Ponemos las label en formato numero en lugar de cadenas de string
def create_ids_4_labels(label):
    global current_id
    id_=0
    if not label in label_ids:
        label_ids[label] = current_id
        current_id +=1
    id_=label_ids[label]
    return id_

array_2_train=[]
label_2_array=[]


#Recorremos el directorio de faces_2_recognize y tenemos las imagenes dentro de las carpetas(label)
for root, dirs, files in os.walk(img_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg") or file.endswith("jpeg"):
            path=os.path.join(root, file)
            label=os.path.basename(root).replace(" ", "_").lower()
            print(path, label)
            pil_image=Image.open(path).convert("L")#lo pasa a escala de grises
            resized_img=pil_image.resize((800,800))
            image_array=np.array(resized_img, "uint8")
            #print(image_array)
            faces=face_classifier.detectMultiScale(image_array)

            for (x, y, w, h) in faces:
                roi_gray = image_array[y:y+h, x:x+w]
                array_2_train.append(np.resize(roi_gray, (450,450)))
                label_2_array.append(create_ids_4_labels(label))
print(label_2_array)
print(label_ids)
#guardamos las asociaciones con pickle
with open("recognizer/labels.pickle", "wb") as f:
    pickle.dump(label_ids, f)

print("Entrenando algoritmo Fisher...")
fisher_recognizer.train(array_2_train, np.array(label_2_array))
print("Entrenando algoritmo LBPHF...")
lbphf_recognizer.train(array_2_train, np.array(label_2_array))

fisher_recognizer.save("recognizer/trainer_fisher.yml")
lbphf_recognizer.save("recognizer/trainer_lbphf.yml")

print("Entrenamientos terminados !! ")

#Con lo que tenemos un yaml para guardar la info que necesita el recognizer para reconocer un rostro

