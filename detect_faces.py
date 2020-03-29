import cv2
from time import sleep
import pickle
import numpy as np

wait_4_camera=True

face_classifier = cv2.CascadeClassifier('resources/classifiers/haarcascade_frontalface_alt2.xml')
eye_classifier = cv2.CascadeClassifier('resources/classifiers/haarcascade_eye.xml')


fisher_recognizer = cv2.face.FisherFaceRecognizer_create()
lbphf_recognizer = cv2.face.LBPHFaceRecognizer_create()

fisher_recognizer.read("recognizer/trainer_fisher.yml")
lbphf_recognizer.read("recognizer/trainer_lbphf.yml")

labels={}

with open("recognizer/labels.pickle", "rb") as f:
	inv_labels=pickle.load(f) 
	labels={v:k for k,v in inv_labels.items()}#invertimos las claves y los valores de posicion

 
while wait_4_camera:
	video_interface = cv2.VideoCapture(0)
	#Si no tenemos camarita...
	if not video_interface.isOpened():
		print('No se detecta cámara, probando de nuevo en 5 segundos!')
		sleep(5)
	else: 
		print('Iniciando sistema de reconocimiento')
		wait_4_camera=False
	
#___________________A partir de aquí se realiza el reconocimiento___________________

while True:
	# Leemos el video frame a frame
	ret, frame = video_interface.read()
	#Lo pasamos a escala de grises
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	#Detectamos las caras
	faces = face_classifier.detectMultiScale(
		gray,
		scaleFactor=1.1,
		minNeighbors=5,
		minSize=(50, 50)
	)

	# Dibujamos el rectangulito en las caras
	for (x, y, w, h) in faces:
		frame=cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 150), 2)
		#RegionOfInterest--->la carita buena
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = frame[y:y+h, x:x+w]

		id_, confidence= fisher_recognizer.predict(np.resize(roi_gray, (450,450)))

		font = cv2.FONT_HERSHEY_SIMPLEX
		cv2.putText(frame, "Fisher->"+labels[id_],(x,y-5),font, 1, (255,255,255),2,cv2.LINE_AA)

		id_, confidence= lbphf_recognizer.predict(np.resize(roi_gray, (200,200)))
		if confidence<100:
			cv2.putText(frame, "LBPHF->"+labels[id_]+"--Confidence: "+str(round(100-confidence)),(x,y-35),font, 1, (255,255,255),2,cv2.LINE_AA)
		else:
			cv2.putText(frame, "LBPHF->Unknown",(x,y-35),font, 1, (255,255,255),2,cv2.LINE_AA)

		eyes = eye_classifier.detectMultiScale(roi_gray, minNeighbors=8, minSize=(50,50))
		for (ex,ey,ew,eh) in eyes:
			cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,0,0),2)

	# Sacamos una ventana con el video 
	cv2.imshow('Video', frame)
	#Rompemos si pretamos 'q'
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# Liberamos la interfaz de video y destruimos las ventanas creadas
print('Saliendo...')
video_interface.release()
cv2.destroyAllWindows()
