from __future__ import division

import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import pygame
import time
import glob
import random
import math

from djitellopy import Tello
from keras.models import Sequential
from keras.backend.tensorflow_backend import set_session
from keras.models import Model, load_model
from keras.layers import Input, Dense, Flatten, AveragePooling2D, MaxPooling2D, Conv2D, Activation, Concatenate, Dropout, BatchNormalization
from keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
from keras.regularizers import l2
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint
from pygame.locals import *

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)


class FaceRecognition(object):
    def __init__(self):
        self.path_face = "N:\\CODE\\DroneAI\\Data2\\face"  # Path of face image for training
        self.path_no_face = "N:\\CODE\\DroneAI\\Data2\\no_face"  # Path of no_face image for training
        self.model_path = "C:\\Users\\phuoc\\OneDrive\\Desktop\\CODE\\DeepLearning\\Model\\FaceRecognition.h5"
        self.total_img = 10000  # Total images used for deep learning
        self.img_size = 100  # Size of the image
        self.channel = 1  # 1 for grayscale and 3 for rgb
        self.num_class = 2
        self.total_training_face = 4000
        self.total_training_no_face = 4000
        self.total_evaluating_face = 1000
        self.total_evaluating_no_face = 1000
        self.regularizer = 0  # 0.000001  # Regularize the kernel_regularizer
        self.batch_size = 50
        self.epoch = 50
        self.steps_per_epoch = int((self.total_training_face + self.total_training_no_face) / self.batch_size)
        self.validation_steps = int((self.total_evaluating_face + self.total_evaluating_no_face) / self.batch_size)
        self.lr = 0.001  # Learning rate
        self.decay = self.lr / (self.steps_per_epoch * self.epoch)  # Decay for learning rate

    def load_image(self, num):
        # Initialize label
        label = np.zeros(self.num_class)

        # Path of the image
        path = self.path_face + str(num) + ".jpg"

        if num < self.total_training_face + self.total_evaluating_face:  # Get the face image
            # Initialize the video
            video = cv2.VideoCapture("N:\\CODE\\DroneAI\\Data\\face.mp4")

            # Get the specific frame in the video
            frame_np = np.random.randint(0, int(video.get(cv2.CAP_PROP_FRAME_COUNT)))  # Get the specific frame in the video
            video.set(1, frame_np)
            ret, frame_vid = video.read()

            # Resize the frame
            frame_vid = cv2.resize(frame_vid, (self.img_size, self.img_size))

            # Save the image
            cv2.imwrite(path, frame_vid)

            # Load data and label
            data = img_to_array(load_img(path, color_mode='grayscale', target_size=(self.img_size, self.img_size)))
            label[0] = 1
        else:  # Get the no_face image
            # Initialize the video
            video = cv2.VideoCapture("N:\\CODE\\DroneAI\\Data\\no_face.mp4")

            # Get the specific frame in the video
            frame_np = np.random.randint(0, int(video.get(cv2.CAP_PROP_FRAME_COUNT)))  # Get the specific frame in the video
            video.set(1, frame_np)
            ret, frame_vid = video.read()

            # Resize the frame
            frame_vid = cv2.resize(frame_vid, (self.img_size, self.img_size))

            # Save the image
            cv2.imwrite(path, frame_vid)

            # Load data and label
            data = img_to_array(load_img(path=path, color_mode='grayscale', target_size=(self.img_size, self.img_size)))
            label[1] = 1

        # Convert data and label into fp32
        data.astype('float32')
        label.astype('float32')

        # Normalize data
        data /= 255.0

        return data, label

    def main(self):
        x_train = np.zeros((self.total_training_face + self.total_training_no_face, self.img_size, self.img_size, self.channel))  # Training data
        y_train = np.zeros((self.total_training_face + self.total_training_no_face, self.num_class))  # Label of training data
        x_test = np.zeros((self.total_evaluating_face + self.total_evaluating_no_face, self.img_size, self.img_size, self.channel))  # Testing data
        y_test = np.zeros((self.total_evaluating_face + self.total_evaluating_no_face, self.num_class))  # Label of testing data
        index = 0
        training_index = 0
        validation_index = 0

        multi_processing = mp.Pool(processes=mp.cpu_count())
        for results in multi_processing.imap(self.load_image, range(self.total_img)):
            if index < self.total_training_face:
                x_train[training_index], y_train[training_index] = results
                training_index += 1
            elif index < self.total_training_face + self.total_evaluating_face:
                x_test[validation_index], y_test[validation_index] = results
                validation_index += 1
            elif index < self.total_training_face + self.total_training_no_face + self.total_evaluating_face:
                x_train[training_index], y_train[training_index] = results
                training_index += 1
            else:
                x_test[validation_index], y_test[validation_index] = results
                validation_index += 1

            index += 1

        print(index)
        print(training_index)
        print(validation_index)
        multi_processing.close()
        multi_processing.join()

        # Build model and data generator
        input_shape = (self.img_size, self.img_size, self.channel)
        model = Sequential()

        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(256, (3, 3), activation='relu'))
        model.add(Conv2D(256, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dense(self.num_class, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=self.lr, momentum=0.0, decay=self.decay, nesterov=False), metrics=['accuracy'])

        generator = ImageDataGenerator(rotation_range=45, width_shift_range=0.1, height_shift_range=0.1, brightness_range=[0.5, 1.5], shear_range=0.2, horizontal_flip=True, vertical_flip=True,
                                       zoom_range=[0.5, 1.5])

        # Create and compile model for deep learning
        r = model.fit_generator(generator=generator.flow(x=x_train, y=y_train, batch_size=self.batch_size), steps_per_epoch=self.steps_per_epoch,
                                validation_data=generator.flow(x=x_test, y=y_test, batch_size=self.batch_size), validation_steps=self.validation_steps, max_queue_size=700,
                                callbacks=[ModelCheckpoint(filepath=self.model_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')], workers=self.batch_size, epochs=self.epoch)

        print(model.summary())

        plt.plot(r.history['loss'], label='loss')
        plt.plot(r.history['val_loss'], label='val_loss')
        plt.legend()
        plt.show()

        plt.plot(r.history['acc'], label='acc')
        plt.plot(r.history['val_acc'], label='val_acc')
        plt.legend()
        plt.show()

    def predict(self):
        model = load_model(self.model_path)
        data = np.zeros((978, self.img_size, self.img_size, self.channel))
        index = 0

        for file in glob.glob("N:\\CODE\\DroneAI\\Data3\\*.jpg"):
            data[index] = img_to_array(load_img(file, color_mode='grayscale', target_size=(self.img_size, self.img_size)))

        data.astype('float32')
        data /= 255.0

        result = model.predict(data)
        print(result)


class DroneFollowTraining(object):
    def __init__(self):
        self.path = "C:\\Users\\phuoc\\OneDrive\\Desktop\\CODE\\DeepLearning\\Model\\DroneFollow.h5"  # Path of the saved model
        self.data_path = "N:\\CODE\\DroneAI\\Data3\\*.jpg"
        self.num_img = 15000  # Total images
        self.train_num = 12000  # Number of images used for training
        self.validation_num = 3000  # Number of images used for validation
        self.height = 90  # The original height of the image is 720. Therefore, reducing the height of image by factor of 8
        self.width = 120  # The original width of the image is 960. Therefore, reducing the width of image by factor of 8
        self.channel = 1  # 1 for grayscale and 3 for rgb
        self.num_class = 7+7+3+1  # Number of information in a label
        self.regularizer = 1e-7  # Regularize the kernel_regularizer
        self.batch_size = 10  # The size of batch
        self.epoch = 10  # Number of epoch
        self.steps_per_epoch = int(self.train_num / self.batch_size)  # Number of step taken per epoch
        self.validation_steps = int(self.validation_num / self.batch_size)  # Number of validation step taken per epoch
        self.lr = 1e-5  # Learning rate
        self.decay = 1e-7  # Decay for learning rate

    def stem(self, x):
        x = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='valid', kernel_regularizer=l2(self.regularizer))(x)
        x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', kernel_regularizer=l2(self.regularizer))(x)
        x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(self.regularizer))(x)

        x1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x)

        x2 = Conv2D(filters=96, kernel_size=(3, 3), strides=(2, 2), padding='valid', kernel_regularizer=l2(self.regularizer))(x)

        x = Concatenate(axis=3)([x1, x2])

        x3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_regularizer=l2(self.regularizer))(x)
        x3 = Conv2D(filters=96, kernel_size=(3, 3), strides=(1, 1), padding='valid', kernel_regularizer=l2(self.regularizer))(x3)

        x4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_regularizer=l2(self.regularizer))(x)
        x4 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', kernel_regularizer=l2(self.regularizer))(x4)
        x4 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', kernel_regularizer=l2(self.regularizer))(x4)
        x4 = Conv2D(filters=96, kernel_size=(3, 3), strides=(1, 1), padding='valid', kernel_regularizer=l2(self.regularizer))(x4)

        x = Concatenate(axis=3)([x3, x4])

        x5 = Conv2D(filters=192, kernel_size=(3, 3), strides=(1, 1), padding='valid', kernel_regularizer=l2(self.regularizer))(x)

        x6 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='valid')(x)

        x = Concatenate(axis=3)([x5, x6])
        x = BatchNormalization(axis=3)(x, training=False)
        x = Activation('relu')(x)

        return x

    def inception_A(self, x):
        x_shortcut = x

        x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_regularizer=l2(self.regularizer))(x)

        x2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_regularizer=l2(self.regularizer))(x)
        x2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(self.regularizer))(x2)

        x3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_regularizer=l2(self.regularizer))(x)
        x3 = Conv2D(filters=48, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(self.regularizer))(x3)
        x3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(self.regularizer))(x3)

        x = Concatenate(axis=3)([x1, x2, x3])
        x = Conv2D(filters=384, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_regularizer=l2(self.regularizer))(x)

        x = Concatenate(axis=3)([x, x_shortcut])
        x = BatchNormalization(axis=3)(x, training=False)
        x = Activation('relu')(x)

        return x

    def inception_B(self, x):
        x_shortcut = x

        x1 = Conv2D(filters=192, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_regularizer=l2(self.regularizer))(x)

        x2 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_regularizer=l2(self.regularizer))(x)
        x2 = Conv2D(filters=160, kernel_size=(1, 7), strides=(1, 1), padding='same', kernel_regularizer=l2(self.regularizer))(x2)
        x2 = Conv2D(filters=192, kernel_size=(7, 1), strides=(1, 1), padding='same', kernel_regularizer=l2(self.regularizer))(x2)

        x = Concatenate(axis=3)([x1, x2])
        x = Conv2D(filters=1154, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_regularizer=l2(self.regularizer))(x)

        x = Concatenate(axis=3)([x, x_shortcut])
        x = BatchNormalization(axis=3)(x, training=False)
        x = Activation('relu')(x)

        return x

    def inception_C(self, x):
        x_shortcut = x

        x1 = Conv2D(filters=192, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_regularizer=l2(self.regularizer))(x)

        x2 = Conv2D(filters=192, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_regularizer=l2(self.regularizer))(x)
        x2 = Conv2D(filters=224, kernel_size=(1, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(self.regularizer))(x2)
        x2 = Conv2D(filters=256, kernel_size=(3, 1), strides=(1, 1), padding='same', kernel_regularizer=l2(self.regularizer))(x2)

        x = Concatenate(axis=3)([x1, x2])
        x = Conv2D(filters=2048, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_regularizer=l2(self.regularizer))(x)

        x = Concatenate(axis=3)([x, x_shortcut])
        x = BatchNormalization(axis=3)(x, training=False)
        x = Activation('relu')(x)

        return x

    def reduction_A(self, x):
        x1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x)

        x2 = Conv2D(filters=384, kernel_size=(3, 3), strides=(2, 2), padding='valid', kernel_regularizer=l2(self.regularizer))(x)

        x3 = Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_regularizer=l2(self.regularizer))(x)
        x3 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(self.regularizer))(x3)
        x3 = Conv2D(filters=384, kernel_size=(3, 3), strides=(2, 2), padding='valid', kernel_regularizer=l2(self.regularizer))(x3)

        x = Concatenate(axis=3)([x1, x2, x3])
        x = BatchNormalization(axis=3)(x, training=False)
        x = Activation('relu')(x)

        return x

    def reduction_B(self, x):
        x1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x)

        x2 = Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_regularizer=l2(self.regularizer))(x)
        x2 = Conv2D(filters=384, kernel_size=(3, 3), strides=(2, 2), padding='valid', kernel_regularizer=l2(self.regularizer))(x2)

        x3 = Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_regularizer=l2(self.regularizer))(x)
        x3 = Conv2D(filters=288, kernel_size=(3, 3), strides=(2, 2), padding='valid', kernel_regularizer=l2(self.regularizer))(x3)

        x4 = Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_regularizer=l2(self.regularizer))(x)
        x4 = Conv2D(filters=288, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(self.regularizer))(x4)
        x4 = Conv2D(filters=320, kernel_size=(3, 3), strides=(2, 2), padding='valid', kernel_regularizer=l2(self.regularizer))(x4)

        x = Concatenate(axis=3)([x1, x2, x3, x4])
        x = BatchNormalization(axis=3)(x, training=False)
        x = Activation('relu')(x)

        return x

    def inception_resnet_v2(self, input_layer):
        x = self.stem(input_layer)

        for i in range(5):
            x = self.inception_A(x)

        x = self.reduction_A(x)

        for i in range(10):
            x = self.inception_B(x)

        x = self.reduction_B(x)

        for i in range(5):
            x = self.inception_C(x)

        x = AveragePooling2D(pool_size=(2, 2), padding='valid')(x)
        x = Dropout(rate=0.2)(x)

        x = Flatten()(x)

        x = Dense(units=512, activation='relu', kernel_regularizer=l2(self.regularizer))(x)
        x = Dense(units=512, activation='relu', kernel_regularizer=l2(self.regularizer))(x)
        x = Dense(self.num_class, activation='sigmoid')(x)

        return x

    def build_model(self):
        input_shape = (self.height, self.width, self.channel)

        x_input = Input(input_shape)
        # x = ZeroPadding2D(padding=(3, 3), data_format='channels_last')(x_input)
        x = self.inception_resnet_v2(x_input)

        model = Model(inputs=x_input, outputs=x)
        print('Model built')

        return model

    # This function is used to deal with imbalance classes
    def generate_label(self):
        path = glob.glob(self.data_path)  # Path to access all the data
        check_label = []  # List of labels of data
        result_path = []  # Path for the same elements
        final_path = []  # Path of data to return

        # Get the labels from all data
        for label in path:
            labels = label.split("Data3")[1].split('.')[0][1:].split('_')
            for_back = labels[2]
            up_down = labels[3]
            yaw = labels[4]
            check_label.append(for_back + '_' + up_down + '_' + yaw)

        unique_label = list(set(check_label))  # Remove all the repeated elements
        class_num = math.ceil(self.num_img / len(unique_label))  # Number of same label allowed. This is used to prevent imbalance of classes

        # Iterate through the list of unique element
        for i in range(len(unique_label)):
            elements = []

            for j in range(len(path)):
                if unique_label[i] == check_label[j]:
                    elements.append(path[j])

            random.shuffle(elements)

            if len(elements) >= class_num:
                while len(elements) >= class_num:
                    elements.pop()
            else:
                while len(elements) < class_num:
                    elements.append(elements[random.randint(0, len(elements) - 1)])

            result_path.append(elements)

        for i in range(len(result_path)):
            for j in range(len(result_path[i])):
                final_path.append(result_path[i][j])

        random.shuffle(final_path)
        return final_path

    def load_image(self):
        check_list = ['-50', '-35', '-20', '0', '20', '35', '50']
        x_train = np.zeros((self.train_num, self.height, self.width, 1))  # Training data
        y_train = np.zeros((self.train_num, self.num_class))  # Label of training data
        x_test = np.zeros((self.validation_num, self.height, self.width, 1))  # Testing data
        y_test = np.zeros((self.validation_num, self.num_class))  # Label of testing data
        training_index = 0
        validation_index = 0
        check_label = []

        path = self.generate_label()

        for i in range(self.num_img):
            print(i)
            label = np.zeros(18)
            check = False
            labels = path[i].split("Data3")[1].split('.')[0][1:].split('_')
            left_right = labels[1]
            for_back = labels[2]
            up_down = labels[3]
            yaw = labels[4]
            index1 = index2 = index3 = 0
            check_label.append(for_back + '_' + up_down + '_' + yaw)

            if left_right == '-1':
                check = True
            else:
                if for_back == '-35':
                    index1 = 0
                elif for_back == '0':
                    index1 = 1
                else:
                    index1 = 2

                for sub_index in check_list:
                    if up_down == sub_index:
                        break
                    index2 += 1

                for sub_index in check_list:
                    if yaw == sub_index:
                        break
                    index3 += 1

            if check is not True:
                label[index1] = 1
                label[index2 + 3] = 1
                label[index3 + 3 + 7] = 1
            else:
                label[self.num_class - 1] = 1

            if i < self.train_num:
                x_train[training_index] = img_to_array(load_img(path=path[i], color_mode='grayscale', target_size=(self.height, self.width)))
                y_train[training_index] = label
                training_index += 1
            else:
                x_test[validation_index] = img_to_array(load_img(path=path[i], color_mode='grayscale', target_size=(self.height, self.width)))
                y_test[validation_index] = label
                validation_index += 1

        import collections

        print(collections.Counter(check_label))
        print(len(list(set(check_label))))

        # Convert data and label into fp32
        x_train = x_train.astype('float32')
        y_train = y_train.astype('float32')
        x_test = x_test.astype('float32')
        y_test = y_test.astype('float32')

        # Normalize data
        x_train = x_train / 255.0
        x_test = x_test / 255.0
        # y_train = to_categorical(y_train)
        # y_test = to_categorical(y_test)

        print(training_index)
        print(validation_index)

        return x_train, y_train, x_test, y_test

    def simple_model(self, num):
        input_shape = (self.height, self.width, self.channel)

        if num == 0:
            self.num_class = 3
        else:
            self.num_class = 7

        # Create and compile model for deep learning
        model = Sequential()

        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(self.num_class, activation='softmax'))

        return model

    def main(self):
        x_train, y_train, x_test, y_test = self.load_image()
        path = "C:\\Users\\phuoc\\OneDrive\\Desktop\\CODE\\DeepLearning\\Model\\DroneFollow.h5"

        # Build model and data generator
        model = self.build_model()  # Build model
        training_gen = ImageDataGenerator(brightness_range=[0.5, 1.5])
        valuating_gen = ImageDataGenerator(brightness_range=[0.5, 1.5])

        # SGD(lr=self.lr, momentum=0.0, decay=self.decay, nesterov=False)
        model.compile(loss='binary_crossentropy', optimizer=Adam(lr=self.lr, amsgrad=True), metrics=['accuracy'])
        r = model.fit_generator(generator=training_gen.flow(x=x_train, y=y_train, batch_size=self.batch_size), steps_per_epoch=self.steps_per_epoch,
                                validation_data=valuating_gen.flow(x=x_test, y=y_test, batch_size=self.batch_size), validation_steps=self.validation_steps, max_queue_size=10,
                                callbacks=[ModelCheckpoint(filepath=path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')], workers=self.batch_size, epochs=self.epoch)

        print(model.summary())

        plt.plot(r.history['acc'], label='acc')
        plt.plot(r.history['val_acc'], label='val_acc')
        plt.legend()
        plt.show()

        plt.plot(r.history['loss'], label='loss')
        plt.plot(r.history['val_loss'], label='val_loss')
        plt.legend()
        plt.show()

    def predict(self):
        model = load_model("C:\\Users\\phuoc\\OneDrive\\Desktop\\CODE\\DeepLearning\\Model\\DroneFollow.h5")
        data = np.zeros((1, self.height, self.width, self.channel))
        check_list = ['-50', '-35', '-20', '0', '20', '35', '50']
        num = 0
        for_back_check = up_down_check = yaw_check = 0.0
        index1_check = index2_check = index3_check = 0
        count = 0

        for path in glob.glob("N:\\CODE\\DroneAI\\Data3\\*.jpg"):
            data[0] = img_to_array(load_img(path=path, color_mode='grayscale', target_size=(self.height, self.width)))
            data[0].astype('float32')
            data[0] = data[0] / 255.0
            index1 = index2 = index3 = 0

            print(count)
            count += 1

            prediction = model.predict(data)

            labels = path.split("Data3")[1].split('.')[0][1:].split('_')
            left_right = labels[1]
            for_back = labels[2]
            up_down = labels[3]
            yaw = labels[4]

            if left_right != '-1':
                for i in range(0, 3):
                    if prediction[0][i] > for_back_check:
                        for_back_check = prediction[0][i]
                        index1_check = i

                for i in range(3, 10):
                    if prediction[0][i] > up_down_check:
                        up_down_check = prediction[0][i]
                        index2_check = i

                for i in range(10, 17):
                    if prediction[0][i] > yaw_check:
                        yaw_check = prediction[0][i]
                        index3_check = i

                if for_back == '-35':
                    index1 = 0
                elif for_back == '0':
                    index1 = 1
                else:
                    index1 = 2

                for sub_index in check_list:
                    if up_down == sub_index:
                        break
                    index2 += 1

                for sub_index in check_list:
                    if yaw == sub_index:
                        break
                    index3 += 1

                print(str(index1_check) + " " + str(index2_check) + " " + str(index3_check))
                print(str(index1) + " " + str(index2) + " " + str(index3))
                print(path)
                input('wait')

            if count == 500:
                input('wait')

                if index1 == index1_check and (index2+3) == index2_check and (index3+7+3) == index3_check:
                    num += 1

        print(num)


class DroneRemoteControl(object):
    def __init__(self):
        # Initialize pygame
        pygame.init()

        self.auto = False

        # Create pygame window
        pygame.display.set_caption("Tello Video Stream")
        self.screen = pygame.display.set_mode([800, 800])

        # Initialize Tello drone
        self.tello = Tello()

        # Initialize speed of the drone
        self.low_speed = 20
        self.medium_speed = 35
        self.high_speed = 50
        self.default_speed = 35
        self.speed = self.default_speed

        # Initialize frames per second
        self.fps = 60

        # Initialize remote control
        self.send_rc_control = False

        # Drone velocities between -100~100
        self.for_back_velocity = 0
        self.left_right_velocity = 0
        self.up_down_velocity = 0
        self.yaw_velocity = 0

        self.path = "C:\\Users\\phuoc\\OneDrive\\Documents\\opencv\\sources\\data\\haarcascades_cuda\\haarcascade_frontalface_alt2.xml"
        self.classifier = cv2.CascadeClassifier(self.path)
        self.model = load_model("C:\\Users\\phuoc\\OneDrive\\Desktop\\CODE\\DeepLearning\\Model\\FaceRecognition.h5")

        # Create update timer
        pygame.time.set_timer(USEREVENT+1, 50)

        self.index = 13000
        self.img_path = "N:\\CODE\\DroneAI\\Data3\\"

        self.tello.connect()

    def run(self):
        print(self.tello.get_battery())

        if not self.tello.connect():
            print('Tello is not connected')
            return

        if not self.tello.streamoff():
            print("Could not stop video stream")
            return

        if not self.tello.streamon():
            print("Could not start video stream")
            return

        frame_read = self.tello.get_frame_read()

        should_stop = False
        while not should_stop:
            for event in pygame.event.get():
                if event.type == USEREVENT + 1:
                    self.update()
                elif event.type == QUIT:
                    should_stop = True
                elif event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        should_stop = True
                    elif event.type == K_t:
                        self.tello.takeoff()
                    elif event.type == K_l:
                        self.tello.land()
                    else:
                        self.key_pressed(event.key)
                elif event.type == KEYUP:
                    self.key_released(event.key)

            if frame_read.stopped:
                frame_read.stop()
                break

            self.screen.fill([0, 0, 0])
            frame = frame_read.frame

            if self.auto is True:
                path = self.img_path + str(self.index) + "_" + str(-1) + "_" + str(-1) + "_" + str(-1) + "_" + str(-1) + ".jpg"
                self.index += 1

                if self.index > 15500:
                    self.tello.land()
                    self.auto = False

                cv2.imwrite(path, frame)
                """
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                detect_face = self.classifier.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=6)

                if len(detect_face) != 0:
                    if detect_face.shape[0] == 1:
                        check = True

                        # for (x, y, w, h) in detect_face:
                        # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        
                        x, y, w, h = detect_face[0]

                        middle_x = x + w/2
                        middle_y = y + h/2

                        if 960 > middle_x > 585:
                            if 960 > middle_x >= 835:
                                self.yaw_velocity = self.high_speed
                            elif 835 > middle_x >= 710:
                                self.yaw_velocity = self.medium_speed
                            else:
                                self.yaw_velocity = self.low_speed
                        elif 0 < middle_x < 375:
                            if 0 < middle_x <= 125:
                                self.yaw_velocity = -self.high_speed
                            elif 125 < middle_x <= 250:
                                self.yaw_velocity = -self.medium_speed
                            else:
                                self.yaw_velocity = -self.low_speed
                        else:
                            self.yaw_velocity = 0

                        if 720 > middle_y > 465:
                            if 720 > middle_y >= 635:
                                self.up_down_velocity = -self.high_speed
                            elif 635 > middle_y >= 550:
                                self.up_down_velocity = -self.medium_speed
                            else:
                                self.up_down_velocity = -self.low_speed
                        elif 0 < middle_y < 255:
                            if 0 < middle_y <= 85:
                                self.up_down_velocity = self.high_speed
                            elif 85 < middle_y <= 170:
                                self.up_down_velocity = self.medium_speed
                            else:
                                self.up_down_velocity = self.low_speed
                        else:
                            self.up_down_velocity = 0

                        if 0 < w < 90 or 0 < h < 90:
                            self.for_back_velocity = self.medium_speed
                        elif 960 > w > 150 or 960 > h > 150:
                            self.for_back_velocity = -self.medium_speed
                        else:
                            self.for_back_velocity = 0

                        path = self.img_path + str(self.index) + "_" + str(self.left_right_velocity) + "_" + str(self.for_back_velocity) + "_" + str(self.up_down_velocity) + "_" + \
                               str(self.yaw_velocity) + ".jpg"
                        self.index += 1

                        if self.index > 15000:
                            self.tello.land()
                            self.auto = False

                        cv2.imwrite(path, frame)
                    else:
                        check = False
                else:
                    check = False

                if check is not True:
                    self.yaw_velocity = 0
                    self.up_down_velocity = 0
                    self.left_right_velocity = 0
                    self.for_back_velocity = 0

                self.tello.send_rc_control(self.left_right_velocity, self.for_back_velocity, self.up_down_velocity, self.yaw_velocity)
                """

            cv2.imshow("Drone", frame)

            # time.sleep(1 / self.fps)

        # Call it always before finishing. I deallocate resources.
        self.tello.end()

    def validate_image(self):
        img_path = "N:\\CODE\\DroneAI\\Data3\\*.jpg"
        left_right_check = 0
        wrong_label = []
        multiple_face = []

        for path in glob.glob(img_path):
            gray_frame = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
            detect_face = self.classifier.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=6)
            labels = str(path).split("Data3")[1].split('.')[0][1:].split('_')
            index, left_right, for_back, up_down, yaw = labels
            print(labels)

            if len(detect_face) != 0:
                if detect_face.shape[0] == 1:
                    x, y, w, h = detect_face[0]
                    middle_x = x + w / 2
                    middle_y = y + h / 2

                    if 960 > middle_x > 585:
                        if 960 > middle_x >= 835:
                            yaw_check = self.high_speed
                        elif 835 > middle_x >= 710:
                            yaw_check = self.medium_speed
                        else:
                            yaw_check = self.low_speed
                    elif 0 < middle_x < 375:
                        if 0 < middle_x <= 125:
                            yaw_check = -self.high_speed
                        elif 125 < middle_x <= 250:
                            yaw_check = -self.medium_speed
                        else:
                            yaw_check = -self.low_speed
                    else:
                        yaw_check = 0

                    if 720 > middle_y > 465:
                        if 720 > middle_y >= 635:
                            up_down_check = -self.high_speed
                        elif 635 > middle_y >= 550:
                            up_down_check = -self.medium_speed
                        else:
                            up_down_check = -self.low_speed
                    elif 0 < middle_y < 255:
                        if 0 < middle_y <= 85:
                            up_down_check = self.high_speed
                        elif 85 < middle_y <= 170:
                            up_down_check = self.medium_speed
                        else:
                            up_down_check = self.low_speed
                    else:
                        up_down_check = 0

                    if 0 < w < 90 or 0 < h < 90:
                        for_back_check = self.medium_speed
                    elif 960 > w > 150 or 960 > h > 150:
                        for_back_check = -self.medium_speed
                    else:
                        for_back_check = 0

                    print(str(left_right_check) + ' ' + str(for_back_check) + ' ' + str(up_down_check) + ' ' + str(yaw_check))

                    if str(yaw) != str(yaw_check) or str(up_down) != str(up_down_check) or str(for_back) != str(for_back_check):
                        wrong_label.append(index)
                else:
                    multiple_face.append(index)

        print(wrong_label)
        print(len(wrong_label))
        print(multiple_face)
        print(len(multiple_face))

    def key_pressed(self, key):
        """ Update velocities based on key pressed
        Arguments:
            key: pygame key
        """
        if key == pygame.K_UP:  # set forward velocity
            self.for_back_velocity = self.speed
        elif key == pygame.K_DOWN:  # set backward velocity
            self.for_back_velocity = -self.speed
        elif key == pygame.K_LEFT:  # set left velocity
            self.left_right_velocity = -self.speed
        elif key == pygame.K_RIGHT:  # set right velocity
            self.left_right_velocity = self.speed
        elif key == pygame.K_w:  # set up velocity
            self.up_down_velocity = self.speed
        elif key == pygame.K_s:  # set down velocity
            self.up_down_velocity = -self.speed
        elif key == pygame.K_a:  # set yaw clockwise velocity
            self.yaw_velocity = -self.speed
        elif key == pygame.K_d:  # set yaw counter clockwise velocity
            self.yaw_velocity = self.speed
        elif key == pygame.K_e:
            if self.auto is not True:
                self.auto = True
            else:
                self.auto = False

    def key_released(self, key):
        """ Update velocities based on key released
        Arguments:
            key: pygame key
        """
        if key == pygame.K_UP or key == pygame.K_DOWN:  # set zero forward/backward velocity
            self.for_back_velocity = 0
        elif key == pygame.K_LEFT or key == pygame.K_RIGHT:  # set zero left/right velocity
            self.left_right_velocity = 0
        elif key == pygame.K_w or key == pygame.K_s:  # set zero up/down velocity
            self.up_down_velocity = 0
        elif key == pygame.K_a or key == pygame.K_d:  # set zero yaw velocity
            self.yaw_velocity = 0
        elif key == pygame.K_t:  # takeoff
            self.tello.takeoff()
            self.send_rc_control = True
        elif key == pygame.K_l:  # land
            self.tello.land()
            self.send_rc_control = False

    def update(self):
        """ Update routine. Send velocities to Tello."""
        if self.send_rc_control:
            self.tello.send_rc_control(self.left_right_velocity, self.for_back_velocity, self.up_down_velocity, self.yaw_velocity)


class DroneFollowInference(object):
    def __init__(self):
        self.model_path = "C:\\Users\\phuoc\\OneDrive\\Desktop\\CODE\\DeepLearning\\Model\\DroneFollow.h5"  # Path of the saved model
        self.img_path = "N:\\CODE\\DroneAI\\Data1\\2.jpg"
        self.img_size = 100  # Size of image
        self.display = 800  # Size of display
        self.channel = 1  # Channel of image
        self.face_prob = 0.5  # Probability of face is detected

    def test(self):
        height = width = size = best = 0
        data = np.zeros((1, self.img_size, self.img_size, self.channel))
        model = load_model(self.model_path)
        data[0] = img_to_array(load_img(path=self.img_path, color_mode='grayscale', target_size=(self.img_size, self.img_size)))
        data[0].astype('float32')
        data[0] = data[0] / 255.0
        results = model.predict(data)

        for i in results:
            print(i)

        # Determine if the frame has face in it
        if results[0][0] > self.face_prob:

            # Determine the height coordinate
            for i in range(1, self.img_size + 1):
                if results[0][i] > best:
                    best = results[0][i]
                    height = i

            best = 0

            # Determine the width coordinate
            for i in range(self.img_size + 1, self.img_size * 2 + 1):
                if results[0][i] > best:
                    best = results[0][i]
                    width = i

            best = 0

            # Determine the size of face
            for i in range(self.img_size * 2 + 1, len(results[0])):
                if results[0][i] > best:
                    best = results[0][i]
                    size = i

        print(height)
        print(width)
        print(size)
        print(height - 1)
        print(width - self.img_size - 1)
        print(size - (self.img_size * 2 + 1))
        print()

    def predict(self):
        cap = cv2.VideoCapture(0)
        model = load_model(self.model_path)
        data = np.zeros((1, self.img_size, self.img_size, self.channel))

        while True:
            # Initialize
            height = width = size = best = 0

            # Capture the frame
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (self.display, self.display), interpolation=cv2.INTER_AREA)

            # Save the frame
            cv2.imwrite(self.img_path, frame)

            # Load data
            data[0] = img_to_array(load_img(path=self.img_path, color_mode='grayscale', target_size=(self.img_size, self.img_size)))
            data[0].astype('float32')
            data[0] = data[0] / 255.0

            # inference the data
            results = model.predict(data)

            # Determine if the frame has face in it
            if results[0][0] > self.face_prob:

                # Determine the height coordinate
                for i in range(1, self.img_size + 1):
                    if results[0][i] > best:
                        best = results[0][i]
                        height = i

                best = 0

                # Determine the width coordinate
                for i in range(self.img_size + 1, self.img_size * 2 + 1):
                    if results[0][i] > best:
                        best = results[0][i]
                        width = i

                best = 0

                # Determine the size of face
                for i in range(self.img_size * 2 + 1, len(results[0])):
                    if results[0][i] > best:
                        best = results[0][i]
                        size = i

            height_coor = int(((height - 1) / self.img_size) * self.display)
            width_coor = int(((width - self.img_size - 1) / self.img_size) * self.display)
            face_size = int(((size - (self.img_size * 2 + 1)) / self.img_size) * self.display)
            cv2.rectangle(frame, (height_coor, width_coor), (height_coor + face_size, width_coor + face_size), (0, 255, 0), 2)
            cv2.imshow("Video", frame)

            # Exit the video by pressing enter key
            if cv2.waitKey(1) == 13:
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    time.sleep(1)
    print('1 for training, 2 for inference, 3 for drone remote control, 4 for face recognition')
    user = input('Enter the command: ')

    if user == '1':
        drone = DroneFollowTraining()
        # drone.main()
        drone.predict()
        print('Success')
    elif user == '2':
        inference = DroneFollowInference()
        # inference.test()
        inference.predict()
        print('Success')
    elif user == '3':
        control = DroneRemoteControl()
        control.run()
        # control.validate_image()
        print('Success')
    elif user == '4':
        face_recognition = FaceRecognition()
        face_recognition.main()
        print('Success')
    else:
        print('Error')
