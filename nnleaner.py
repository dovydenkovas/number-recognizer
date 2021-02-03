""" Программа для обучения нейронной сети. """
import keras
from keras.datasets import mnist
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.models import Sequential
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    answer = input("Вы уверены, что хотите заново обучить нейронную сеть? [да/Нет]")
    if answer == "да":
        print("Start")
        class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ]

        (train_X, train_Y), (test_X, test_Y) = mnist.load_data()

        train_X = train_X.reshape(-1, 28, 28, 1)
        test_X = test_X.reshape(-1, 28, 28, 1)

        train_X = train_X.astype('float32')
        test_X = test_X.astype('float32')
        train_X = train_X / 255
        test_X = test_X / 255

        train_Y_one_hot = to_categorical(train_Y)
        test_Y_one_hot = to_categorical(test_Y)

        model = Sequential([
            Conv2D(64, (8, 8), input_shape=(28, 28, 1), activation='relu'),
            MaxPooling2D(pool_size=(3, 3)),
            Conv2D(64, (3, 3), activation='tanh'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(128, activation='tanh'),
            Dropout(0.25),
            Dense(28, activation='tanh'),
            Dropout(0.55),
            Dense(10, activation='softmax'),
        ])

        model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),
                      metrics=['accuracy'])

        model.fit(train_X, train_Y_one_hot, batch_size=120, epochs=5)

        test_loss, test_acc = model.evaluate(test_X, test_Y_one_hot)
        print('Test loss', test_loss)
        print('Test accuracy', test_acc)

        predictions = model.predict(test_X)

        # def plot_image(i, predictions_array, true_label, img):
        #     predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
        #     plt.grid(False)
        #     plt.xticks([])
        #     plt.yticks([])
        #
        #     plt.imshow(img, cmap=plt.cm.binary)
        #
        #     predicted_label = np.argmax(predictions_array)
        #     if predicted_label == true_label:
        #         color = 'blue'
        #     else:
        #         color = 'red'
        #
        #     plt.xlabel(class_names[predicted_label], color=color)
        #
        #
        # num_rows = 10
        # num_cols = 25
        # num_images = num_rows*num_cols
        # plt.figure(figsize=(2*2*num_cols, 2*num_rows))
        # for i in range(num_images):
        #   plt.subplot(num_rows, 2*num_cols, 2*i+1)
        #   plot_image(i, predictions, test_Y, test_X)
        # plt.show()

        model.save("numbers.model")
