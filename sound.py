import cv2
import numpy as np
import os
#import simpleaudio as sa





#wave_obj = sa.WaveObject.from_wave_file("warrior.wav")
#wave_obj.play()
video=cv2.VideoCapture('warrior.mp4')
with open('testing.txt', 'r') as f:
    a=f.read()
    
while (video.isOpened() and ((a == 'do start') or (a == 'do start\n'))):
    with open('testing.txt', 'r') as f:
        a=f.read()
    grabbed, frame=video.read()
        
        
    cv2.namedWindow ('key', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty ('key', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('key',frame)
    cv2.waitKey(1) 
        
        
        
video.release()
cv2.destroyAllWindows()



