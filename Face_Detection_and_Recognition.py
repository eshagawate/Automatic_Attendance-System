import cv2, numpy, os
import csv
from datetime import datetime
import numpy as np
import time

haar_file = 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)
datasets = r'C:\Users\eshag\OneDrive\Desktop\Projects\Face Recog\code\dataset'
print('Training...')
(images, labels, names, id) = ([], [], {}, 0)

for (subdirs, dirs, files) in os.walk(datasets):
    for subdir in dirs:
        names[id] = subdir
        subjectpath = os.path.join(datasets, subdir)
        for filename in os.listdir(subjectpath):
            path = subjectpath + '/' + filename
            label = id
            images.append(cv2.imread(path, 0))
            labels.append(int(label))
        id +=1

(images, labels) = [numpy.array(lis) for lis in [images, labels]]
print(images, labels)                   
(width, height) = (256, 256)
model = cv2.face.LBPHFaceRecognizer_create()
#model =  cv2.face.FisherFaceRecognizer_create()

model.train(images, labels)
print("Training Complete")
webcam = cv2.VideoCapture(0)
time.sleep(5.0)
cv2.waitKey(1)

def markAttendance(name):
    with open(r'student.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            time_now = datetime.now()
            tString = time_now.strftime('%H:%M:%S')
            dString = time_now.strftime('%d/%m/%Y')
            f.writelines(f'\n{name},{tString},{dString}')

while True:
    (_, im) = webcam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(im,(x,y),(x+w,y+h),(255,255,0),2)
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (width, height))
        cnt=0
        prediction = model.predict(face_resize)
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3)
        if prediction[1]>400:
            cv2.putText(im,'%s - %.0f' % (names[prediction[0]],prediction[1]),(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,2,(0, 0, 255))
            name=names[prediction[0]]
            print (names[prediction[0]])
            cnt=0
            markAttendance(name)
            
            # open the file in the write mode
            #with open(r'student.csv', 'w') as f:
            # create the csv writer
                #writer = csv.writer(f, delimiter=',')
            # write a row to the csv file
                #d  = datetime.now()
                #for item in range(1000):
                    #writer.writerow([names[prediction[0]],d])
                
        else:
            cnt+=1
            cv2.putText(im,'Unknown',(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
            if(cnt>100):
                print("Unknown Person")
                cv2.imwrite("unKnown.jpg",im)
                cnt=0
    cv2.imshow('FaceRecognition', im)
    key = cv2.waitKey(10)
    if key == 27:
        break

webcam.release()
cv2.destroyAllWindows()



