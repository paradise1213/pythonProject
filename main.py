
import cv2
import numpy as np
import face_recognition

imgFar = face_recognition.load_image_file("Images/far.jpg")
imgFar = cv2.cvtColor(imgFar,cv2.COLOR_BGR2RGB)

imgTest = face_recognition.load_image_file("Images/farTest.jpg")
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

#finding faces

faceLoc=face_recognition.face_locations(imgFar)[0]
encodeFar=face_recognition.face_encodings(imgFar)[0]
cv2.rectangle(imgFar,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

faceLocTest=face_recognition.face_locations(imgTest)[0]
encodeTest=face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)

results = face_recognition.compare_faces([encodeFar],encodeTest)
faceDis = face_recognition.face_distance([encodeFar],encodeTest)

print(results, faceDis)
cv2.putText(imgTest,f'{results}{round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv2.imshow("far", imgFar)
cv2.imshow("farTest", imgTest)
cv2.waitKey(0)

