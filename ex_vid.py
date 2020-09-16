import cv2
import time


#time.sleep(56)

cap=cv2.VideoCapture('/usr/local/src/trt_pose/tasks/human_pose/vid.mp4')

while (cap.isOpened()):
    ret,frame=cap.read()
    cv2.namedWindow ('key', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty ('key', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('key',frame)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

