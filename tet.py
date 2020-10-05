from threading import Thread
import os
import socket
import time
import select
import sys


HOST = '192.168.1.113' # Enter IP or Hostname of your server
PORT = 1024 # Pick an open Port (1000+ recommended), must match the server port
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST,PORT))

def ecoute():
    
    while True:
        
        reply = s.recv(1024)
        reply = reply.decode()
        with open('testing.txt', 'w') as f:
            f.write('%s' % reply)
            
       
def execute():
    #temp = 1
    while True:
        with open('testing.txt', 'r') as f:
            a=f.read()   
        
        if (a == 'do start'):
            os.system("python3 detect_video_opt.py")
            


def send():
    t=time.time()
    while True:
        
        if (time.time()-t)>2:
            with open('aghlat.txt', 'r') as f:
                b=f.read() 
            print(b)
            s.send(b.encode('utf-8'))
            t=time.time()
        #print('sending')


if __name__ == "__main__":
    t1 = Thread(target = ecoute)
    t2 = Thread(target = execute)
    t3 = Thread(target = send)
    t1.setDaemon(True)
    t2.setDaemon(True)
    t3.setDaemon(True)
    t1.start()
    t2.start()
    t3.start()
    while True:
        pass

