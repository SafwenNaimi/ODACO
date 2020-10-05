import socket
import subprocess
import sys
import time
import os

HOST = '192.168.1.113' # Enter IP or Hostname of your server
PORT = 1024 # Pick an open Port (1000+ recommended), must match the server port
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST,PORT))

reply = s.recv(1024)
reply = reply.decode()
while True:
    #command = input('Enter your command: ')
    #s.send(command.encode('utf-8'))
    reply = s.recv(1024)
    reply = reply.decode()
    while reply == 'do start':
        reply = s.recv(1024)
        reply = reply.decode()
        
        os.system('python3 detect_video_opt.py')
        print(reply)
        #subprocess.call("python3 detect_video_opt.py", shell=True)
        if reply == 'do stop':
            sys.exit()
       
        
    while reply == 'do stop':
        print(reply)
        sys.exit()
