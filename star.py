from threading import Thread
import socket
import os
import cv2
import time

img = cv2.imread("/usr/local/src/trt_pose/tasks/human_pose/start.png")
cv2.namedWindow('key',cv2.WINDOW_NORMAL)        # Create a named window
cv2.setWindowProperty ('key', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.imshow('key',img)
cv2.waitKey(0)

#cv2.destroyAllWindows()
 
