from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, BatchNormalization, Activation
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from tensorflow.python.client import device_lib
import numpy as np
from keras.utils import np_utils
import warnings
from keras import backend as K

warnings.filterwarnings("ignore")


def traincnn(data):
    print(device_lib.list_local_devices())
    print(K.tensorflow_backend._get_available_gpus())
    exit(0)
    number_of_classes = 2
    model = Sequential()

    model.add(Conv2D(128, (5, 5), padding='same', activation='relu', input_shape=(128, 128, 1)))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(Conv2D(384, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(384, (3, 3), padding='same', activation='relu'))

    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(5120, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(5120, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(number_of_classes, activation='softmax'))

    print(model.summary())

    fulldata = []
    fulllabels = []
    for claasvalue, images in data[0].items():
        for im in images:
            fulldata.append(im)
            fulllabels.append(claasvalue)

    arrays = [im for im in fulldata]
    fulldata = np.stack(arrays, axis=0)

    fulllabels = np.array(fulllabels)
    fulllabels = np_utils.to_categorical(fulllabels, number_of_classes)

    batch_size = 256
    epochs = 100
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(fulldata, fulllabels, batch_size=batch_size, epochs=epochs, verbose=1,
                        validation_data=(fulldata, fulllabels))

    return [model, history]


def testcnn(model):

    plt.figure(figsize=[8, 6])
    plt.plot(model[1].history['loss'], 'r', linewidth=3.0)
    plt.plot(model[1].history['val_loss'], 'b', linewidth=3.0)
    plt.legend(['Training loss', 'Validation Loss'], fontsize=18)
    plt.xlabel('Epochs ', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.title('Loss Curves', fontsize=16)

    plt.figure(figsize=[8, 6])
    plt.plot(model[1].history['acc'], 'r', linewidth=3.0)
    plt.plot(model[1].history['val_acc'], 'b', linewidth=3.0)
    plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=18)
    plt.xlabel('Epochs ', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.title('Accuracy Curves', fontsize=16)
