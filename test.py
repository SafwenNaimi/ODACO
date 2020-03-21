import datetime
from PIL import Image
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.python.keras.models import Model
from collections import deque
start = datetime.datetime.now()

mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")
Q = deque(maxlen=20)

class_labels=['Angry','Happy','Neutral','Sad','Surprise']
model=tf.keras.models.load_model('model.h5')




frame = cv2.imread('face.jpg')
#gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)


#for (x,y,w,h) in faces:
        #cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        #roi_gray = gray[y:y+h,x:x+w]
#roi_color=gray[y:y+h,x:x+w]
roi_gray = cv2.resize(frame,(128,128),interpolation=cv2.INTER_AREA)




if np.sum([roi_gray])!=0: # If in the frame there is at least one face
    roi = roi_gray.astype('float')/255.0 #Normalizing the frame that we are getting from webcam
    roi = img_to_array(roi)
    roi = np.expand_dims(roi,axis=0) #This function expands the array by inserting a new axis at the specified position
            #roi=roi.reshape((-1,128,128,3))


        # make a prediction on the ROI, then lookup the class


    preds = model.predict(roi)[0]

    Q.append(preds)

    results = np.array(Q).mean(axis=0)


    i = np.argmax(results)

    label = class_labels[i]


            #if i==2:
end = datetime.datetime.now()
elapsed = end - start
print(label, elapsed)