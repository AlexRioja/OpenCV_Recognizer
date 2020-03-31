import cv2
import os
import argparse

# argparse para traer el valor de los parametros de entrada
ap = argparse.ArgumentParser()
ap.add_argument("-l", "--label", required=True,
	help="Etiqueta que poner a las caras recogidas por el script")
ap.add_argument("-c", "--crop",action='store_true',
	help="Utilize este parámetro cuando quiera que se genere un dataset con las porciones de imagen de la cara unicamente.")
args = vars(ap.parse_args())

face_classifier = cv2.CascadeClassifier('resources/classifiers/haarcascade_frontalface_alt2.xml')
eye_classifier = cv2.CascadeClassifier('resources/classifiers/haarcascade_eye.xml')


try:
    os.mkdir("resources/faces_2_recognize/"+args['label'])
except OSError as error:
    pass
if args['crop']:
	print("[INFO] El dataset se generará con los recortes de las caras")
else:
	print("[INFO] Usa el parámetro -c para un dataset con crop de las caras")

video_interface = cv2.VideoCapture(0)
n=0
#vamos a crear datasets de por ejemplo 50 fotos :D

def image_preprocessing(img):
	#ampliamos el espectro de la imagen para normalizar la intesidad de pixel (mayor contraste, mejor extracción de features)
	return cv2.equalizeHist(img)

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
			w_rn= int(-0.1*w/2)
			roi_color = frame[y:y+h, x:x+w]
			roi_gray = gray[y:y+h,x+w_rn:x+w-w_rn]
			font = cv2.FONT_HERSHEY_SIMPLEX
			
			cv2.putText(frame, "Carita detectada :)",(x,y-5),font, 1, (255,255,255),2,cv2.LINE_AA)

			#cv2.imwrite("resources/faces_2_recognize/"+label+"/"+str(n)+".jpg", gray[y-50:y+h+50, x-50:x+w+50])
			if args['crop']:
				#pre_procesamos la imagen para generar un buen dataset
				processed_img=image_preprocessing(roi_gray)
				if processed_img.shape < (50,50):
					processed_img=cv2.resize(processed_img, (50,50),interpolation=cv2.INTER_AREA)
				else:
					processed_img=cv2.resize(processed_img, (50,50),interpolation=cv2.INTER_CUBIC)
			else:
				#pre_procesamos la imagen para generar un buen dataset
				processed_img=image_preprocessing(gray)

			cv2.imwrite("resources/faces_2_recognize/"+args['label']+"/"+str(n)+".jpg", processed_img)

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
