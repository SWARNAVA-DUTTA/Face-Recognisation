
import cv2
import pickle


face_cascade = cv2.CascadeClassifier('C:/Users/SWARNAVA/Desktop/facerecognition/cascade/data/haarcascade_frontalface_alt2.xml')
recognizer=cv2.face.LBPHFaceRecognizer_create()
recognizer.read("C://Users//SWARNAVA//Desktop//facerecognition//trainer.yml")
v = cv2.VideoCapture(0)

labels={"person_name":1}
with open("C://Users//SWARNAVA//Desktop//facerecognition//labels.pickle",'rb') as f:
    og_labels=pickle.load(f)
    labels={v:k for k,v in og_labels.items()}


while (True):
    ret, frame = v.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (x, y, w, h) in faces:
        #print(x, y, w, h)
        roi_gray = gray[y:y+h,x:x+w]
        roi_color = frame[y:y+h,x:x+w]

        id,conf =recognizer.predict(roi_gray)
        if conf>=15 and conf<=85:
            print(id)
            print(labels[id])
            font = cv2.FONT_HERSHEY_COMPLEX
            name = labels[id]
            color = (255,255,71)
            stroke = 2
            cv2.putText(frame,name,(x,y),font,2,color,stroke,cv2.LINE_AA)

        img_item = "face1.png"
        cv2.imwrite(img_item,roi_gray)

        color = (255,165,215)
        s = 2
        end_xcord = x+w
        end_ycord = y+h
        cv2.rectangle(frame,(x,y),(end_xcord,end_ycord),color,s)

    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

v.release()
cv2.destroyAllWindows()