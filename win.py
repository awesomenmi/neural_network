import sys
import cv2
import predict
from PIL import Image
import face_recognition
from PyQt5 import QtCore, QtGui 
from PyQt5.QtGui import QIcon 
from PyQt5 import QtWidgets
from PyQt5.uic import loadUi


class win(QtWidgets.QDialog):
    def __init__(self):
        super(win,self).__init__()
        loadUi('un.ui', self)
        self.setWindowTitle('Face ID')
        self.setWindowIcon(QIcon('2.png'))
        self.Button1.clicked.connect(self.loadClicked)
        self.Button2.clicked.connect(self.predictClicked)
    @QtCore.pyqtSlot()

    def loadClicked(self):
        fname, filter = QtWidgets.QFileDialog.getOpenFileName(self, 'Открыть файл','D:\\diplom\\', "Image Files (*.jpg; *.jpeg)")
        #print(fname)
        if fname:
            self.loadImage(fname)
            self.displayFace(fname)

    @QtCore.pyqtSlot()

    def predictClicked(self):
        self.label44.setText(predict.predict())

    def loadImage(self,fname):
        self.image = cv2.imread(fname)
        self.displayImage()

    def displayImage(self):
       
       qformat = QtGui.QImage.Format_Indexed8
        
       if len(self.image.shape)==3:
           if(self.image.shape[2])==4:
               qformat = QtGui.QImage.Format_RGBA8888
           else:
               qformat = QtGui.QImage.Format_RGB888
        
       img = QtGui.QImage(self.image,self.image.shape[1],self.image.shape[0], self.image.strides[0],qformat)
       img = img.rgbSwapped()
       self.imgLabel.setPixmap(QtGui.QPixmap.fromImage(img))
       self.imgLabel.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)
       self.imgLabel.setScaledContents(True)

    def displayFace(self,fname):
        
       image = face_recognition.load_image_file(fname)
       face_locations = face_recognition.face_locations(image)
       
       #print("I found {} face(s) in this photograph.".format(len(face_locations)))
       
       for face_location in face_locations:
           top, right, bottom, left = face_location
           #print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))
           
           face_image = image[top:bottom, left:right]
           image = Image.fromarray(face_image)

       image.save("face.jpg")

       self.image = cv2.imread(r'face.jpg')

       qformat = QtGui.QImage.Format_Indexed8
        
       if len(self.image.shape)==3:
           if(self.image.shape[2])==4:
               qformat = QtGui.QImage.Format_RGBA8888
           else:
               qformat = QtGui.QImage.Format_RGB888
        
       img = QtGui.QImage(self.image,self.image.shape[1],self.image.shape[0], self.image.strides[0],qformat)
       img = img.rgbSwapped()
       self.imgLabel2.setPixmap(QtGui.QPixmap.fromImage(img))
       self.imgLabel2.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)
       self.imgLabel2.setScaledContents(True)

       self.label33.setText("Верхний левый угол: {}, {}".format(top, left))
       self.label34.setText("Нижний правый угол: {}, {}".format(bottom, right))

       
app = QtWidgets.QApplication(sys.argv)
window = win()
window.show()

sys.exit(app.exec_())