import cv2
import numpy as np
import os







video=cv2.VideoCapture('tree_pose.mp4')
with open('testing.txt', 'r') as f:
    a=f.read()
    
while (video.isOpened() and ((a == 'do start') or (a == 'do start\n'))):
    with open('testing.txt', 'r') as f:
        a=f.read()
    grabbed, frame=video.read()
        
        
    cv2.namedWindow ('key', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty ('key', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('key',frame)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
        
        
video.release()
cv2.destroyAllWindows()



