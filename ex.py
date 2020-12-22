import sys
import os
import PyQt5
from PyQt5.QtWidgets import (QWidget, QProgressBar, QLabel, QPushButton, QApplication, QTextEdit, QComboBox, QMessageBox)
from PyQt5.QtCore import QBasicTimer
from PyQt5.QtGui import QPixmap
from PyQt5.QtGui import QColor
key = 0
key_two = 0
network = 1

class Example(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()
    def initUI(self):
        self.setAutoFillBackground(True)
        p = self.palette()
        p.setColor(self.backgroundRole(), PyQt5.QtGui.QColor(188,143,143))
        self.setPalette(p)
        self.pbar = QProgressBar(self)
        self.pbar.setGeometry(50, 400, 350, 25)
        self.btn = QPushButton('Распознавание', self)
        self.btn.resize(180,40)
        self.btn.move(130, 220)
        self.btn.clicked.connect(self.model_nn)

        self.btn = QPushButton('Обучение', self)
        self.btn.resize(180,40)
        self.btn.move(130, 300)
        self.btn.clicked.connect(self.doAction)
        os.listdir(name)
		for img_path in os.listdir(r"\home\ksenia\diplom\picture_nn"):
            if key==0:
                pixmap1 = QPixmap(r"\home\ksenia\diplom\picture_nn\"+img_path)
                pixmap1=pixmap1.scaled(300,450,PyQt5.QtCore.Qt.KeepAspectRatio)
                lbl1 = QLabel(self)
                lbl1.setPixmap(pixmap1)
                lbl1.move(410,100)
                key=1


        self.timer = QBasicTimer()
        self.step = 0
        self.name = QLabel("Процент выполнения обучения", self)
        self.name.move(100, 380)
        #self.name.TextFlag(0x0200)
        self.name = QLabel("Модель сверточной нейронной сети", self)
        self.name.move(80, 80)
        self.name = QLabel("Процент правильного распознавания:", self)
        self.name.move(80, 460)

        self.text_per=QTextEdit(self)
        if network ==1:
        self.text_per.setText("94,31%")
                elif network==2:
                self.text_per.setText("92,53%")
                self.text_per.resize(70,30)
                self.text_per.move(180,500)
                self.text_per.show()
        #self.lbl = QLabel("Сетьс 9 скрытымислоями, чередование 2-1", self)
        combo = QComboBox(self)
        combo.addItems(["Сетьс 9 скрытымислоями, чередование 2-1", "Сетьс 8 скрытымислоями, чередование 1-1"], [network=1,network=2])
                
        combo.resize(350, 30)
        combo.move(50, 120)
        #self.lbl.move(50, 150)

        combo.activated[str].connect(self.onActivated)
        self.setGeometry(100, 100, 1080, 570)
        self.setWindowTitle('Система распознавания образов на основе сверточных нейронных сетей')
        self.show()

    def closeEvent(self, event):

        reply = QMessageBox.question(self, 'Предупреждение', "Вы уверены, что хотите закрыть приложение?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()



    def timerEvent(self, e):

        if self.step >= 100:
            self.timer.stop()
            self.btn.setText('Обучение')
            return

        self.step = self.step + 1
        self.pbar.setValue(self.step)
		
	def model_nn(self)
	    if key_two==0:
		    class_n=loaded_model.predict_nn(network)
			key_two=1
		if key_two==1
		    key=0
			key_two=0
		for n in range(14):
		    if class_n[a]==1:
			    pixmap2 = QPixmap(r"\home\ksenia\diplom\sample\"+str(n)+".jpg")
                pixmap2=pixmap2.scaled(300,300,PyQt5.QtCore.Qt.KeepAspectRatio )
                lbl2 = QLabel(self)
                lbl2.setPixmap(pixmap2)
                lbl2.move(730,100)

                pixmap3 = QPixmap(r"\home\ksenia\diplom\sample\"+str(n)+"2.jpg")
                pixmap3=pixmap3.scaled(300,84,PyQt5.QtCore.Qt.KeepAspectRatio )
                lbl3 = QLabel(self)
                lbl3.setPixmap(pixmap3)
                lbl3.move(730,430)    
		


    def doAction(self):

        if self.timer.isActive():
		    if network==1:
			    self.timer.stop(train_neural_network_1. train_pictures())
                self.btn.setText('Обучение-в процессе')
				
		else network==2:
			    self.timer.stop(train_neural_network_2. train_pictures())
                self.btn.setText('Обучение-в процессе')

else:
            if network==1:
			    self.timer.start(100,train_neural_network_1. train_pictures())
				scores=train_neural_network_1. train_pictures()
				
		    else network==2:
			    self.timer.start(100, train_neural_network_2. train_pictures())
				scores=train_neural_network_2. train_pictures())
            self.btn.setText('Пауза')
			self.text_per.clean()
			self.text_per.setText(scores,"%")



    def onActivated(self, text):

        self.lbl.setText(text)
        self.lbl.adjustSize()

if __name__ == '__main__':

    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())
