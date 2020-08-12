import cv2
import sys
import time
print('Treating The Results.. Please Wait')
for i in range(0,5):
    
    sys.stdout.write(str(i)+' ')
    sys.stdout.flush()
    time.sleep(1)


image=cv2.imread('saf.jpg')

cv2.imshow('out',image)
cv2.waitKey(0)
