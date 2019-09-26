"""
Predefined keras models.
"""

from keras.layers import Conv2D, MaxPooling2D, Flatten, BatchNormalization, Activation
from keras.models import Sequential
from keras.regularizers import l2


def build_lenet5_conv(input_shape, kernel_initializer="glorot_uniform"):
    """
    Convolution from LeNet-5 (described here: https://engmrk.com/lenet-5-a-classic-cnn-architecture/)

    For the MNIST classification task, it may be followed by a fully-connected layer of size 120 then by a size 84 fully connected layer.

    :param input_shape:
    :return:
    """
    model = Sequential()
    model.add(
        Conv2D(6, (5, 5), padding='valid', activation='relu', kernel_initializer=kernel_initializer, input_shape=input_shape))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Conv2D(16, (5, 5), padding='valid', activation='relu', kernel_initializer=kernel_initializer))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    return model


def build_vgg19_conv(input_shape, kernel_initializer="glorot_uniform", weight_decay=0.0001):
    """
    Convolution from vgg19 (described here: https://github.com/BIGBALLON/cifar-10-cnn/blob/master/3_Vgg19_Network/Vgg19_keras.py)

    For the cifar10 classification task, it may be followed by two fully connected layers of size 4096 (very big).

    :param input_shape:
    :param kernel_initializer:
    :param weight_decay:
    :return:
    """
    model = Sequential()

    # Block 1
    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(weight_decay),
                     kernel_initializer=kernel_initializer, name='block1_conv1', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(weight_decay),
                     kernel_initializer=kernel_initializer, name='block1_conv2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

    # Block 2
    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(weight_decay),
                     kernel_initializer=kernel_initializer, name='block2_conv1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(weight_decay),
                     kernel_initializer=kernel_initializer, name='block2_conv2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

    # Block 3
    model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=l2(weight_decay),
                     kernel_initializer=kernel_initializer, name='block3_conv1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=l2(weight_decay),
                     kernel_initializer=kernel_initializer, name='block3_conv2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=l2(weight_decay),
                     kernel_initializer=kernel_initializer, name='block3_conv3'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=l2(weight_decay),
                     kernel_initializer=kernel_initializer, name='block3_conv4'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

    # Block 4
    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=l2(weight_decay),
                     kernel_initializer=kernel_initializer, name='block4_conv1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=l2(weight_decay),
                     kernel_initializer=kernel_initializer, name='block4_conv2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=l2(weight_decay),
                     kernel_initializer=kernel_initializer, name='block4_conv3'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=l2(weight_decay),
                     kernel_initializer=kernel_initializer, name='block4_conv4'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))

    # Block 5
    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=l2(weight_decay),
                     kernel_initializer=kernel_initializer, name='block5_conv1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=l2(weight_decay),
                     kernel_initializer=kernel_initializer, name='block5_conv2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=l2(weight_decay),
                     kernel_initializer=kernel_initializer, name='block5_conv3'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=l2(weight_decay),
                     kernel_initializer=kernel_initializer, name='block5_conv4'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))

    model.add(Flatten(name='flatten'))

    return model

