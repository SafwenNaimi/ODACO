import cv2
import numpy as np
import imutils
import sys
import time

keyboard = np.zeros((1000, 1000, 3), np.uint8)

image_0=cv2.imread('postures-de-yoga8.jpg')
image_1=cv2.imread('a1c78be1f0ba682b2a8a9df496be33d7.jpg')

#print(image_0.shape)
#print(image_1.shape)

key=np.vstack([image_0,image_1])
#print(key.shape)
#print(keyboard.shape)
keyboard = cv2.resize(keyboard, (2500, 1190))
final=np.hstack([key,keyboard])
final = imutils.resize(final, height=600)
cv2.putText(final,"Levez les bras et le torse d une seule respiration. ",(500,150),1,2,(255,0,0),2)
cv2.putText(final,"Le pied arriere devrait marquer un angle de 90 degres. ",(500,200),1,2,(255,0,0),2)
cv2.putText(final,"la cuisse avant devrait etre parallele au sol. ",(500,250),1,2,(255,0,0),2)
cv2.putText(final,"l avant du genou directement au-dessus de la cheville. ",(500,300),1,2,(255,0,0),2)
cv2.putText(final,"Reprenez la posture de l autre cote du corps. ",(500,350),1,2,(255,0,0),2)

cv2.imshow("Result",final)
#cv2.imwrite("hh.png",final)
cv2.waitKey(0)


for i in range(0,5):
    
    sys.stdout.write(str(i)+' ')
    sys.stdout.flush()
    time.sleep(1)
