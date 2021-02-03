""" Инструменты для управления нейронной сетью. """

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Отключает сообщения tensorflow о наличие/отсутствии видеокарты.

from tensorflow import keras
import numpy as np

model = None


def load():
    """
    Загружает модель нейронной сети из папки numbers.model в глобальную переменную model.
    :return: True если загрузка успешна, False если при загрузке возникли ошибки.
    """
    global model
    print("Загрузка нейронной сети... ", end="")
    try:
        model = keras.models.load_model("models/numbers.model")
    except OSError:
        print('Fail')
        return False
    print("OK")
    return True


def get_value(image):
    """
    Используя нейронную сеть numbers.model распознает цифру.
    :param image: массив np.array (28,28)
    :return: Распознанную цифру и вероятность распознования.
    """

    arr = np.zeros((28, 28))
    for i in range(len(image)):
        for j in range(len(image[0])):
            arr[i][j] = 1-image[i][j][0]/255
    arr = np.array([arr]).reshape(-1,28,28,1)

    predict = model.predict(arr)
    argmax = np.argmax(predict)
    return argmax, predict[0][argmax]
