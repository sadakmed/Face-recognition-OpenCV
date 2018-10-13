import os 
import matplotlib.pyplot as plt
import pickle
from PIL import Image
import numpy as np 
import cv2

x_train=[]
y_labels=[]
labels={'makavelli':1,'maynard':2,'serj':3}
detector = cv2.CascadeClassifier('/home/akura/opencv-master/data/haarcascades/haarcascade_frontalface_alt.xml')
recognizor=cv2.face.LBPHFaceRecognizer_create()

for root,dirs,files in os.walk('dataset'):
    for file in files:
            path=os.path.join(root,file)
            label=root.split('/')[1]
            img=Image.open(path).convert("L")
            imgarr=np.array(img,'uint8')
            faces=detector.detectMultiScale(imgarr)
            for (x,y,w,h) in faces:
                x_train.append(imgarr[y:y+h,x:x+w])
                y_labels.append(labels[label])     
               


recognizor.train(x_train,np.array(y_labels))
recognizor.save('model.yml')