import socket
import subprocess


HOST = '192.168.1.113' # Enter IP or Hostname of your server
PORT = 1024 # Pick an open Port (1000+ recommended), must match the server port
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST,PORT))

#Lets loop awaiting for your input
while True:
    #command = input('Enter your command: ')
    #s.send(command.encode('utf-8'))
    reply = s.recv(1024)
    reply=reply.decode()
    if reply == 'do start':
        
        subprocess.call(" python3 detect_video.py ", shell=True)
    #elif reply == 'do stop':

        
