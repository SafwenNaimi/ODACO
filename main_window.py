# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'test.ui'
#
# Created by: PyQt5 UI code generator 5.14.2
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

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()
    Form.showFullScreen()
    QtCore.QTimer.singleShot(5000, Form.close)
    sys.exit(app.exec_())
