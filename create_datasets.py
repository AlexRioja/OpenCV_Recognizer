import cv2                  
import numpy as np         
import sys
import os

face_classifier = cv2.CascadeClassifier('resources/classifiers/haarcascade_frontalface_alt2.xml')
eye_classifier = cv2.CascadeClassifier('resources/classifiers/haarcascade_eye.xml')

label=sys.argv[1]
try: 
    os.mkdir("resources/faces_2_recognize/"+label) 
except OSError as error: 
    print(error) 


video_interface = cv2.VideoCapture(0)
n=0
#vamos a crear datasets de por ejemplo 50 fotos :D

while n<50:
	# Leemos el video frame a frame
	ret, frame = video_interface.read()
	#Lo pasamos a escala de grises
	if ret:
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		#Detectamos las caras
		faces = face_classifier.detectMultiScale(
			gray,
			scaleFactor=1.1,
			minNeighbors=5,
			minSize=(50, 50)
		)

		for (x, y, w, h) in faces:
			frame=cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 150), 2)

			roi_color = frame[y:y+h, x:x+w]
			font = cv2.FONT_HERSHEY_SIMPLEX

			cv2.putText(frame, "Carita detectada :)",(x,y-5),font, 1, (255,255,255),2,cv2.LINE_AA)

			cv2.imwrite("resources/faces_2_recognize/"+label+"/"+str(n)+".jpg", gray[y-50:y+h+50, x-50:x+w+50])
			#cv2.imwrite("resources/faces_2_recognize/"+label+"/"+str(n)+".jpg", gray)
			cv2.waitKey(400) #Capturamos una cara cada X ms
			cv2.imshow('Caritas detectadas', roi_color)
			n+=1

		cv2.imshow('Video WebCam', frame) 
		#Rompemos si pretamos 'q'
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
# Liberamos la interfaz de video y destruimos las ventanas creadas
print('Saliendo...')
video_interface.release()
cv2.destroyAllWindows()
