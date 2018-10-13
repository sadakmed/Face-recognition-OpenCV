import cv2 
import numpy as np 
labels=np.array(['sadak','maynard','serj'])
detector = cv2.CascadeClassifier('/home/akura/opencv-master/data/haarcascades/haarcascade_frontalface_alt.xml')
recognizor=cv2.face.LBPHFaceRecognizer_create()
recognizor.read('model.yml')
cap=cv2.VideoCapture(0)


while True:
    ref,frame = cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=detector.detectMultiScale(gray)
    for (x,y,w,h) in faces:
        smface=gray[y:y+h,x:x+w]
        label,predic=recognizor.predict(smface)
        if predic > 30 : 
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,25,25),2)
            cv2.putText(frame,"{}".format(labels[label-1]),(x,y),cv2.FONT_HERSHEY_COMPLEX,0.5,(20,200,20),1,cv2.LINE_AA)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()