"""  Основной файл программы для перевода ручного ввода цифр в их текстовое представление.
Файл содержит графический интерфейс (классы DrawingField и Window) и точку запуска программы.
"""

import sys

import numpy as np
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QPainter, QColor, QPen, QFont, QImage
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QWidget, QPushButton,  QLineEdit, QMessageBox
from qimage2ndarray import recarray_view

import nn


class DrawingField(QWidget):
    """
    Виджет - Поле для рисования.
    Имеет заданый размер (256, 256). Виджет создан на основе QWidget и QImage.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.drawing = False
        self.image = QImage(256, 256, QImage.Format_RGB32)
        self.image.fill(Qt.white)
        self.last_point = None
        self.line_size = 16

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.last_point = event.pos()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False

    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.LeftButton) & self.drawing:
            painter = QPainter(self.image)
            painter.setPen(QPen(Qt.black, self.line_size, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.drawLine(self.last_point, event.pos())
            self.last_point = event.pos()
            self.update()

    def paintEvent(self, event):
        canvasPainter = QPainter(self)
        canvasPainter.drawImage(QPoint(0, 0), self.image)


class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Распознаватель цифр')
        self.setFixedSize(630, 420)

        self.info = QLabel(self, text="Вероятность: -")
        self.info.setFont(QFont("", 14))
        self.info.setGeometry(20, 330, 600, 50)

        self.input = DrawingField(self)
        self.input.setGeometry(20, 80, 256, 256)

        self.equal_lbl = QLabel(self, text="=")
        self.equal_lbl.setFont(QFont("", 100))
        self.equal_lbl.setGeometry(280, 140, 100, 100)

        self.output = QLineEdit(self)
        self.output.setFont(QFont("", 160))
        self.output.setGeometry(360, 80, 256, 256)

        self.clear_button = QPushButton(self, text="Очистить поле")
        self.clear_button.setGeometry(20, 10, 256, 50)
        self.clear_button.clicked.connect(self.clear_input)

        self.translate_button = QPushButton(self, text="Перевести в текст")
        self.translate_button.setGeometry(360, 10, 256, 50)
        self.translate_button.clicked.connect(self.translate_number)

        self.show()

    def clear_input(self):
        self.input.image.fill(Qt.white)
        self.repaint()

    def translate_number(self):
        img = np.array(recarray_view(self.input.image.scaled(28, 28)))
        prediction, percent = nn.get_value(img)
        self.info.setText(f'Вероятность: {round(percent*1000)/10}%')
        self.output.setText(str(prediction))


if __name__ == "__main__":
    app = QApplication([])
    if not nn.load():
        QMessageBox.warning(None, "Ошибка",
                            "При загрузке нейронной сети возникла ошибка. Пожалуйста, проверьте её наличие.")
    else:
        window = Window()
    sys.exit(app.exec_())
