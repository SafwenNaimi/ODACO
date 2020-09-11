# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'tes.ui'
#
# Created by: PyQt5 UI code generator 5.10.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

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
        self.label_2.setGeometry(QtCore.QRect(-20, 0, 1971, 1101))
        self.label_2.setText("")
        self.label_2.setPixmap(QtGui.QPixmap("/usr/local/src/trt_pose/tasks/human_pose/SAF.png"))
        self.label_2.setScaledContents(True)
        self.label_2.setObjectName("label_2")

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))


if __name__ == "__main__":
    import sys
    import time
    time.sleep(28)
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()
    QtCore.QTimer.singleShot(30000, Form.close)
    Form.showFullScreen()
    sys.exit(app.exec_())

