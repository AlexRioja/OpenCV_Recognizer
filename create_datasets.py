import cv2                  
import numpy as np         


face_classifier = cv2.CascadeClassifier('resources/classifiers/haarcascade_frontalface_alt2.xml')
eye_classifier = cv2.CascadeClassifier('resources/classifiers/haarcascade_eye.xml')

video_interface = cv2.VideoCapture(0)
n=0
#vamos a crear datasets de por ejemplo 50 fotos :D

while n<50:
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
    