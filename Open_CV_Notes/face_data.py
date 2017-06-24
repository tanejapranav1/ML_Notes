
# Collect face data

import numpy as np
import cv2

cam = cv2.VideoCapture(0)

#Classifier contains the features of the face
facec = cv2.CascadeClassifier('/home/pranav/perceptron/ML_Notes/Open_CV_Notes/Haar_Classifier/haarcascade_frontalface_default.xml')
# cam1 = cv2.VideoCapture(1)

data = []
ix = 0

#Infinite Loop
while True:
	#type(ret) = bool cam is connected to cv2 or not, fr has the frame 
	ret, fr = cam.read()
	if ret == True:
		# BGR2GRAY and not BRG2GRAY
		gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
		faces = facec.detectMultiScale(gray, 1.3, 5)
		
		for (x,y,w,h) in faces:
			fc = fr[y:y+h, x:x+w, :]
			
			#Resize the photo to 50x50
			r = cv2.resize(fc, (50, 50))

			#For every 10th frame we are storing the frame and To avoid adding more than 20 frames in the list
			if ix%10 == 0 and len(data)<20:
				data.append(r)

			#Drawing a rectangle of (x, y, w, h) and color it, width
			cv2.rectangle(fr,(x,y),(x+w,y+h),(0,0,255),2)
	
		# Show img wrt fr
		cv2.imshow('fr', fr)
		
		#To avoid adding more than 20 frames in the list
		if cv2.waitKey(1) == 27 or len(data)>=20:
		# if cv2.waitkey(1) & 0xFF == ord('q'):
			break
	else:
		print "error"
		break
		
cv2.destroyAllWindows()
data = np.asarray(data)

print data.shape	# Sanity check
np.save('face_02', data)

# Tip: Do sanity checks after writing a small code and try to comment more
