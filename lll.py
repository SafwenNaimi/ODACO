from threading import Thread
import subprocess
import os
import socket
import time
import select
import sys
import cv2
import sys
import time
from PyQt5 import QtCore, QtGui, QtWidgets



HOST = '192.168.1.113' # Enter IP or Hostname of your server
PORT = 1024 # Pick an open Port (1000+ recommended), must match the server port
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST,PORT))
#p1 = subprocess.Popen("python inst.py", shell=False)
#p2 = subprocess.Popen("python prog.py", shell=False)
#pid1 = p1.pid
#pid2 = p2.pid

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1381, 950)
        self.label = QtWidgets.QLabel(Form)
        self.label.setGeometry(QtCore.QRect(-500, -470, 2851, 1831))
        self.label.setText("")
        self.label.setPixmap(QtGui.QPixmap("../Screenshot_2020-08-21 45 pdf - 45 pdf pdf.png"))
        self.label.setScaledContents(True)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(Form)
        self.label_2.setGeometry(QtCore.QRect(0, -700, 3071, 2481))
        self.label_2.setText("")
        
        with open('testing.txt', 'r') as f:
            a=f.read()
        print(a)  
        if a=='do start':
            self.label_2.setPixmap(QtGui.QPixmap("/usr/local/src/trt_pose/tasks/human_pose/prog.png"))
        else:
            self.label_2.setPixmap(QtGui.QPixmap("/usr/local/src/trt_pose/tasks/human_pose/start.png"))
        self.label_2.setObjectName("label_2")

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))


def ecoute():
    
    while True:
        

        reply = s.recv(1024)
        reply = reply.decode()
        with open('testing.txt', 'w') as f:
            f.write('%s' % reply)
            
def tr():
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()
    #Form.showFullScreen()
    #QtCore.QTimer.singleShot(20000, Form.close)
    
    sys.exit(app.exec_())       




if __name__ == "__main__":
    t1 = Thread(target = ecoute)
    t2 = Thread(target = tr)
    
    t1.setDaemon(True)
    t2.setDaemon(True)
    
    t1.start()
    t2.start()
    
    
    
        
    #time.sleep(10)
            
    
    
    while True:
        pass



