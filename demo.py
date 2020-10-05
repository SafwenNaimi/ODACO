from threading import Thread
import socket
import os
import cv2
import time
from pygame import mixer
import random 

HOST = '192.168.1.113' 
PORT = 1024 
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST,PORT))
time.sleep(3)

def show(path):
    img = cv2.imread(path)     
    cv2.namedWindow('key',cv2.WINDOW_NORMAL)        
    cv2.setWindowProperty ('key', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)    
    cv2.imshow('key', img)
    

def increment (x):
    if x == 0:
        a = 1
    if x == 1:
        a = 2
    if x == 2:
        a = 0
    return(a)

def increment_menu (x):
    if x == 0:
        a = 1
    if x == 1:
        a = 2
    if x == 2:
        a = 0
    
    return(a)

def decrement_menu (x):
    if x == 0:
        a = 2
    if x == 1:
        a = 0
    if x == 2:
        a = 1
    
    return(a)

def decrement (x):
    if x == 0:
        a = 2
    if x == 1:
        a = 0
    if x == 2:
        a = 1
    return(a)

def speak(path):
    mixer.init()
    mixer.music.load(path)
    mixer.music.play()


def play(a):
    if a==2:
        
        os.system("(python3 /usr/local/src/trt_pose/tasks/human_pose/sound.py & python3 /usr/local/src/trt_pose/tasks/human_pose/detect_video_opt.py) ; python3 resulte.py")      
    if a==1:
        
        os.system("(python3 /usr/local/src/trt_pose/tasks/human_pose/treez.py & python3 /usr/local/src/trt_pose/tasks/human_pose/ex_pose_tree.py) ; python3 resulte.py") 
    if a==0:
        
        os.system("(python3 /usr/local/src/trt_pose/tasks/human_pose/squa.py & python3 /usr/local/src/trt_pose/tasks/human_pose/ex_squat.py)  ; python3 resulte.py")

"""
def motivate(ch):
    trois = ["trois_lettres_1" , "trois_lettres_2"]
    quatre = ["quatre_lettres_1","quatre_lettres_2"]
    zero = ["zero_lettres_1", "zero_lettres_2","zero_lettres_3"]
    if len(ch)==1:
        if (ch == 'a'):
            speak("/usr/local/src/trt_pose/tasks/human_pose/motivation/bras_droite.mp3")
        if (ch == 'b'):
            speak("/usr/local/src/trt_pose/tasks/human_pose/motivation/bras_gauche.mp3")
        if (ch == 'c'):
            speak("motivation/jambe_gauche.mp3")
        if (ch == 'd'):
            speak("/usr/local/src/trt_pose/tasks/human_pose/motivation/jambe_droite.mp3")
        

    elif len(ch)==2:
        if (ch == 'ab') or (ch == 'ba'):
            speak("/usr/local/src/trt_pose/tasks/human_pose/motivation/partie_superieure.mp3")
        elif (ch == 'cd') or (ch == 'dc'):
            speak("/usr/local/src/trt_pose/tasks/human_pose/motivation/partie_inferieure.mp3")
        else:
            a = random.randrange(1)
            motivate(ch[a])
    elif len(ch)==3:
        speak("/usr/local/src/trt_pose/tasks/human_pose/motivation/"+random.choice(trois)+".mp3")
    elif len(ch) == 4:
        speak("/usr/local/src/trt_pose/tasks/human_pose/motivation/"+random.choice(quatre)+".mp3")
    elif len(ch) == 0:
        speak("/usr/local/src/trt_pose/tasks/human_pose/motivation/"+random.choice(zero)+".mp3")
"""

def motivate(ch):
    trois = ["trois_lettres_1" , "trois_lettres_2"]
    quatre = ["quatre_lettres_1","quatre_lettres_2"]
    zero = ["zero_lettres_1", "zero_lettres_2","zero_lettres_3"]
    if len(ch)==1:
        if (ch == 'a'):
            speak("/usr/local/src/trt_pose/tasks/human_pose/motivation/bras_droite.mp3")
        if (ch == 'b'):
            speak("/usr/local/src/trt_pose/tasks/human_pose/motivation/bras_gauche.mp3")
        if (ch == 'c'):
            speak("motivation/jambe_gauche.mp3")
        if (ch == 'd'):
            speak("/usr/local/src/trt_pose/tasks/human_pose/motivation/jambe_droite.mp3")
        if (ch == 'j'):
            speak("/usr/local/src/trt_pose/tasks/human_pose/motivation/"+random.choice(zero)+".mp3")
        

    elif len(ch)==2:
        if (ch == 'ab') or (ch == 'ba'):
            speak("/usr/local/src/trt_pose/tasks/human_pose/motivation/partie_superieure.mp3")
        elif (ch == 'cd') or (ch == 'dc'):
            speak("/usr/local/src/trt_pose/tasks/human_pose/motivation/partie_inferieure.mp3")
        else:
            a = random.randrange(1)
            motivate(ch[a])
    elif len(ch)==3:
        speak("/usr/local/src/trt_pose/tasks/human_pose/motivation/"+random.choice(trois)+".mp3")
    elif len(ch) == 4:
        speak("/usr/local/src/trt_pose/tasks/human_pose/motivation/"+random.choice(quatre)+".mp3")
    #elif len(ch) == 0:
        #speak("/usr/local/src/trt_pose/tasks/human_pose/motivation/"+random.choice(zero)+".mp3")
def ok():
    
    speak("/usr/local/src/trt_pose/tasks/human_pose/motivation/displayvoc.mp3")
    

def start():
    
    speak("/usr/local/src/trt_pose/tasks/human_pose/motivation/startvoc.mp3")
    
    
def stop():
    
    speak("/usr/local/src/trt_pose/tasks/human_pose/motivation/stopvoc.mp3")
    
    
def menu():
    
    speak("/usr/local/src/trt_pose/tasks/human_pose/motivation/menu.mp3")
    
def backe():
    
    speak("/usr/local/src/trt_pose/tasks/human_pose/motivation/interface_prec.mp3")
    

def back():
    
    speak("/usr/local/src/trt_pose/tasks/human_pose/motivation/backvoc.mp3")

def nexte():
    
    speak("/usr/local/src/trt_pose/tasks/human_pose/motivation/interface_suiv.mp3")

def next():
    
    speak("/usr/local/src/trt_pose/tasks/human_pose/motivation/nextvoc.mp3")
   
def ecoute():  
    while True:
        var = s.recv(1024)
        var = var.decode()
        with open("/usr/local/src/trt_pose/tasks/human_pose/testing.txt", "w") as f:
            f.write("%s" % var)
        

def display():
    etat = ['1', '2', '3']
    etat_menu = ['1', '2', '3']
    actuel = 0
    actuel_menu = 0
    out_menu = False
    while (True):
        #print(out_menu)
        with open("/usr/local/src/trt_pose/tasks/human_pose/testing.txt", "r") as f:
            a=f.read()
       
        print(a)
         
        
        if (a == "do next") or (a == "do next\n"):
            
            if (out_menu==True):
                next()          
                actuel= increment(actuel)
                cv2.destroyAllWindows()
                show("/usr/local/src/trt_pose/tasks/human_pose/images_"+etat[actuel]+".png")
                cv2.waitKey(1000)
            elif (out_menu==False):
                nexte()
                actuel_menu= increment_menu(actuel_menu)
                cv2.destroyAllWindows()
                show("/usr/local/src/trt_pose/tasks/human_pose/images_menu_"+etat_menu[actuel_menu-1]+".png")
                cv2.waitKey(1000)

        elif (a == "do back") or (a == "do back\n"):
            
            if (out_menu==True):
                back()          
                actuel= decrement(actuel)
                cv2.destroyAllWindows()
                show("/usr/local/src/trt_pose/tasks/human_pose/images_"+etat[actuel]+".png")
                cv2.waitKey(1000)
            elif (out_menu==False):
                backe()
                actuel_menu= decrement_menu(actuel_menu)
                cv2.destroyAllWindows()
                show("/usr/local/src/trt_pose/tasks/human_pose/images_menu_"+etat_menu[actuel_menu-1]+".png")
                cv2.waitKey(1000)    
                          
        elif (a == 'do ok') or (a == 'ok\n'):
            ok()
            out_menu=True
            cv2.destroyAllWindows()
            show("/usr/local/src/trt_pose/tasks/human_pose/images_1.png")
            cv2.waitKey(1000)
        elif (a == 'do menu') or (a == 'menu\n'):
            out_menu=False
            cv2.destroyAllWindows()
            show("/usr/local/src/trt_pose/tasks/human_pose/images_menu_"+etat_menu[actuel_menu]+".png")
            cv2.waitKey(1000)
        elif (a == 'do stop') or (a == 'do stop\n'):
            stop()
            if (out_menu==False):
                actuel_menu = 0
                show("/usr/local/src/trt_pose/tasks/human_pose/images_menu_1.png")           
                cv2.waitKey(1000)
            elif (out_menu==True):
                actuel = 0
                show("/usr/local/src/trt_pose/tasks/human_pose/images_1.png")           
                cv2.waitKey(1000)
        elif (a=='do start') or (a=='do start\n'):
            start()
           
            play(actuel)

        with open("/usr/local/src/trt_pose/tasks/human_pose/testing.txt", "w") as f:
                f.write('%s' % '')

def send():
    t=time.time()
    t_1=time.time()
    while True:
        
        if (time.time()-t)>3:
            with open('/usr/local/src/trt_pose/tasks/human_pose/aghlat.txt', 'r') as f:
                b=f.read()
                s.send(b.encode('utf-8'))
                
                with open('/usr/local/src/trt_pose/tasks/human_pose/aghlat.txt', 'w') as f:
                    f.write('%s' % '')
                t=time.time()


        if (time.time()-t_1)>5:
            with open('/usr/local/src/trt_pose/tasks/human_pose/aghlat.txt', 'r') as f:
                 b=f.read()
                 b=(b.replace('*', ''))
                 print(b)
                 motivate(b)
                 
                 t_1=time.time()
            
           


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

