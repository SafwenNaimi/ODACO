import cv2
import os

def show(path):
    img = cv2.imread(path)     
    cv2.namedWindow('key',cv2.WINDOW_NORMAL)        
    cv2.setWindowProperty ('key', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)    
    cv2.imshow('key', img)


with open("/usr/local/src/trt_pose/tasks/human_pose/filename.txt", "r") as f:
    a=f.read()
    #print(a)

    if len(a)==0:
        cv2.destroyAllWindows()
        show("/usr/local/src/trt_pose/tasks/human_pose/result0.png")
        cv2.waitKey(5000)
    elif len(a)==1:
        cv2.destroyAllWindows()
        show("/usr/local/src/trt_pose/tasks/human_pose/result1.png")
        cv2.waitKey(5000)
    elif len(a)==2:
        cv2.destroyAllWindows()
        show("/usr/local/src/trt_pose/tasks/human_pose/result2.png")
        cv2.waitKey(5000)
    elif len(a)==3:
        cv2.destroyAllWindows()
        show("/usr/local/src/trt_pose/tasks/human_pose/result3.png")
        cv2.waitKey(5000)
    elif len(a)==4:
        cv2.destroyAllWindows()
        show("/usr/local/src/trt_pose/tasks/human_pose/result4.png")
        cv2.waitKey(5000)



