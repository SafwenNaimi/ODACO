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

start = datetime.datetime.now()

class_labels=['CORONA','NOT CORONA']
#model=tf.keras.models.load_model('model.h5')

interpreter = tf.lite.Interpreter(model_path="converted_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
#mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")
#Q = deque(maxlen=20)
path='C:/Users/Safwen/Desktop/person373_bacteria_1708.jpeg'
#img = Image.open('NORMAL2-IM-1427-0001.jpeg')
img = load_img(path, target_size=(128, 128))
#print(img.shape())
img = img_to_array(img)
#img = img.resize((128,128))
#plt.imshow(img / 255.)
img = preprocess_input(np.expand_dims(img.copy(), axis=0))
#pred = model.predict(x)[0]
#print(pred)
#pred_class = pred.argmax(axis=-1)
#pred_class=np.argmax(pred)
#print(pred[0][0].round(4))
#print(class_labels[pred_class])

#img = np.reshape(img,[-1,128,128,3])
img=np.interp(img, (img.min(), img.max()), (0, +1))
input_data = np.array(img, dtype=np.float32)

interpreter.set_tensor(input_details[0]['index'],input_data)


interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]['index'])
for pred in output_data:
    print(pred)
    result = np.argmax(pred)
label=class_labels[result]
end = datetime.datetime.now()
elapsed = end - start
print(result,elapsed)
roi=cv2.imread(path)
roi=cv2.resize(roi,(700,700))
cv2.putText(roi,label,(100,100),1,2,(0,255,0),3)
cv2.putText(roi,str(np.max(pred)),(100,150),1,2,(0,255,0),3)
cv2.imshow('imag',roi)
cv2.waitKey(0)


