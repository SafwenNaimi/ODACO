from threading import Thread
import socket
import os
import cv2
import time


HOST = '192.168.1.113' # Enter IP or Hostname of your server
PORT = 1024 # Pick an open Port (1000+ recommended), must match the server port
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST,PORT))


def show(path):
    #print(path)
    img = cv2.imread(path)
    #winname = "Test"
    #cv2.namedWindow ('key', cv2.WINDOW_NORMAL)
    
    cv2.namedWindow('key',cv2.WINDOW_NORMAL)        # Create a named window
    cv2.setWindowProperty ('key', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    #cv2.moveWindow(winname, 500,0)  # Move it to (40,30)
    cv2.imshow('key', img)
    

def increment (x):
    if x == 0:
        a = 1
    if x == 1:
        a = 2
    if x == 2:
        a = 0
    return(a)

def play(a):
    if a==1:
        
        os.system("python3 ex_vid.py & python3 detect_video_opt.py")       
   
def ecoute():  
    while True:
        var = s.recv(1024)
        var = var.decode()
        with open('testing.txt', 'w') as f:
            f.write('%s' % var)

def display():
    etat = ['1', '2', '3']
    actuel = 0
    while (True):
        with open('testing.txt', 'r') as f:
            a=f.read()
         
        if (a == 'do next'):
            
                          
            actuel = increment(actuel)
           
            with open('testing.txt', 'w') as f:
                f.write('%s' % '')
            
            cv2.destroyAllWindows()
            show("/usr/local/src/trt_pose/tasks/human_pose/images_"+etat[actuel]+".png")
            cv2.waitKey(1000)
        elif (a == 'do stop'):
            
            actuel = 0
            show("/usr/local/src/trt_pose/tasks/human_pose/images_1.png")           
            cv2.waitKey(1000)
        elif (a=='do start'):
           
            play(actuel)

def send():
    t=time.time()
    while True:
        
        if (time.time()-t)>2:
            with open('aghlat.txt', 'r') as f:
                b=f.read() 
        
                s.send(b.encode('utf-8'))
            
                t=time.time()

            
           


if __name__ == "__main__":
    t1 = Thread(target = ecoute)
    t2 = Thread(target = display)
    t3 = Thread(target = send)
    t1.setDaemon(True)
    t2.setDaemon(True)
    t3.setDaemon(True)
    t1.start()
    t2.start()
    t3.start()
    while True:
        pass

