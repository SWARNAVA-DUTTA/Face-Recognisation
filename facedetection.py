import cv2
import os
import numpy as np
from PIL import Image
import pickle
import time

face_cascade = cv2.CascadeClassifier('C://Users//SWARNAVA//Desktop//facerecognition//cascade//data//haarcascade_frontalface_alt2.xml')

dir=os.path.dirname(os.path.abspath(__file__))
image= os.path.join(dir,"C://Users//SWARNAVA//Desktop//facerecognition//trainingdata")
name=input("Enter your name:\n")
recognizer=cv2.face.LBPHFaceRecognizer_create()
c_id=0
label_id={}
x_train=[]
y_labels =[]


class Frame:
    def Splitframe(self):
        vidcap = cv2.VideoCapture('C://Users//SWARNAVA//Desktop//facerecognition//Output.avi')
        count = 0
        success = True
        while success:
            success, image = vidcap.read()
            cv2.imwrite("C://Users//SWARNAVA//Desktop//facerecognition//trainingdata//frame%d.jpg" % count, image)
            print('Read a new frame: ', success)
            count += 1
            if success==False:
                try:
                    os.remove("C://Users//SWARNAVA//Desktop//facerecognition//trainingdata//frame%d.jpg" % (count-1))
                except:
                    pass
        else:
            pass

    def Timer(self):
        # The duration in seconds of the video captured
        capture_duration = 10

        cap = cv2.VideoCapture(0)

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('C://Users//SWARNAVA//Desktop//facerecognition//Output.avi', fourcc, 30.0, (640, 480))

        start_time = time.time()
        while (int(time.time() - start_time) < capture_duration):
            ret, frame = cap.read()
            if ret == True:
                frame = cv2.flip(frame, 180)
                out.write(frame)
                cv2.imshow('frame', frame)
            else:
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()


f=Frame()
f.Timer()
f.Splitframe()


for root,dirs,files in os.walk(image):
    for file in files:
        if file.endswith("jpg"):
            path = os.path.join(root, file)
            label = name.replace(" ","-").lower()
            print(label,path)

            #y_labels.append(label)
            #x_train.append(path)
            if not label in label_id:
                label_id[label] = c_id
                c_id+=1
            id=label_id[label]
            print(label_id)
            pil_image = Image.open(path).convert("L")
            size = (550,550)
            final_img = pil_image.resize(size,Image.ANTIALIAS)
            array = np.array(final_img,"uint8")
            print(array)
            faces= face_cascade.detectMultiScale(array, scaleFactor=1.5, minNeighbors=5)
            for (x, y, w, h) in faces:
                roi = array[y:y + h, x:x + w]
                x_train.append(roi)
                y_labels.append(id)
#print(x_train)
#print(y_labels)

with open("C://Users//SWARNAVA//Desktop//facerecognition//labels.pickle","wb") as p:
    pickle.dump(label_id,p)

recognizer.train(x_train,np.array(y_labels))
recognizer.save("C://Users//SWARNAVA//Desktop//facerecognition//trainer.yml")
