
# import os
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# import numpy as np
# import sys
# import subprocess
# import argparse
# import tensorflow as tf
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, Activation, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, ZeroPadding2D, Add
# from tensorflow.keras.optimizers import Adam, SGD
# from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
# from tensorflow.keras import backend as K
#
# import matplotlib.pyplot as plt
# import csv
#
# from utils import *
# from data import load_train_data
#
# # Set image data format
# tf.keras.backend.set_image_data_format('channels_last')
#
# # ----- paths setting -----
# data_path = sys.argv[1] + "/"
# model_path = data_path + "models/"
# log_path = data_path + "logs/"
#
# # ----- params for training and testing -----
# batch_size = 1
# cur_fold = sys.argv[2]
# plane = sys.argv[3]
# epoch = int(sys.argv[4])
# init_lr = float(sys.argv[5])
#
# # ----- Dice Coefficient and cost function for training -----
# smooth = 1.0
#
# def dice_coef(y_true, y_pred):
#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)
#     intersection = K.sum(y_true_f * y_pred_f)
#     return (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
#
# def dice_coef_loss(y_true, y_pred):
#     return -dice_coef(y_true, y_pred)
#
# def get_unet(img_shape, flt=64, pool_size=(2, 2)):
#     """Build and compile Neural Network"""
#     print("Start building NN")
#     inputs = Input(img_shape)
#
#     conv1 = Conv2D(flt, (3, 3), activation='relu', padding='same')(inputs)
#     conv1 = Conv2D(flt, (3, 3), activation='relu', padding='same')(conv1)
#     pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
#
#     conv2 = Conv2D(flt * 2, (3, 3), activation='relu', padding='same')(pool1)
#     conv2 = Conv2D(flt * 2, (3, 3), activation='relu', padding='same')(conv2)
#     pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
#
#     conv3 = Conv2D(flt * 4, (3, 3), activation='relu', padding='same')(pool2)
#     conv3 = Conv2D(flt * 4, (3, 3), activation='relu', padding='same')(conv3)
#     pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
#
#     conv4 = Conv2D(flt * 8, (3, 3), activation='relu', padding='same')(pool3)
#     conv4 = Conv2D(flt * 8, (3, 3), activation='relu', padding='same')(conv4)
#     pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
#
#     conv5 = Conv2D(flt * 16, (3, 3), activation='relu', padding='same')(pool4)
#     conv5 = Conv2D(flt * 16, (3, 3), activation='relu', padding='same')(conv5)
#
#     up6 = concatenate([Conv2DTranspose(flt * 8, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
#     conv6 = Conv2D(flt * 8, (3, 3), activation='relu', padding='same')(up6)
#     conv6 = Conv2D(flt * 8, (3, 3), activation='relu', padding='same')(conv6)
#
#     up7 = concatenate([Conv2DTranspose(flt * 4, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
#     conv7 = Conv2D(flt * 4, (3, 3), activation='relu', padding='same')(up7)
#     conv7 = Conv2D(flt * 4, (3, 3), activation='relu', padding='same')(conv7)
#
#     up8 = concatenate([Conv2DTranspose(flt * 2, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
#     conv8 = Conv2D(flt * 2, (3, 3), activation='relu', padding='same')(up8)
#     conv8 = Conv2D(flt * 2, (3, 3), activation='relu', padding='same')(conv8)
#
#     up9 = concatenate([Conv2DTranspose(flt, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
#     conv9 = Conv2D(flt, (3, 3), activation='relu', padding='same')(up9)
#     conv9 = Conv2D(flt, (3, 3), activation='relu', padding='same')(conv9)
#
#     conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)
#
#     model = Model(inputs=[inputs], outputs=[conv10])
#
#     model.compile(optimizer=Adam(learning_rate=init_lr), loss=dice_coef_loss, metrics=[dice_coef])
#
#     return model
#
# def train(fold, plane, batch_size, nb_epoch):
#     """
#     Train a U-Net model with data from load_train_data()
#
#     Parameters
#     ----------
#     fold : string
#         Which fold is experimenting in 4-fold. It should be one of 0/1/2/3
#
#     plane : char
#         Which plane is experimenting. It is from 'X'/'Y'/'Z'
#
#     batch_size : int
#         Size of mini-batch
#
#     nb_epoch : int
#         Number of epochs to train NN
#
#     init_lr : float
#         Initial learning rate
#     """
#     print("Number of epochs: ", nb_epoch)
#     print("Learning rate: ", init_lr)
#
#     # --------------------- load and preprocess training data -----------------
#     print('-' * 80)
#     print('         Loading and preprocessing train data...')
#     print('-' * 80)
#
#     images_train, masks_train = load_train_data(fold, plane)
#
#     images_row = images_train.shape[1]
#     images_col = images_train.shape[2]
#
#     images_train = preprocess(images_train)
#     masks_train = preprocess(masks_train)
#
#     images_train = images_train.astype('float32')
#     masks_train = masks_train.astype('float32')
#
#     # ---------------------- Create, compile, and train model ------------------------
#     print('-' * 80)
#     print('        Creating and compiling model...')
#     print('-' * 80)
#
#     model = get_unet((images_row, images_col, 1), pool_size=(2, 2))
#     print(model.summary())
#
#     print('-' * 80)
#     print('        Fitting model...')
#     print('-' * 80)
#
#     ver = 'unet_fd{}_{}_ep{}_lr{}.csv'.format(cur_fold, plane, epoch, init_lr)
#     csv_logger = CSVLogger(log_path + ver)
#     model_checkpoint = ModelCheckpoint(model_path + ver + ".keras",
#                                        monitor='loss',
#                                        save_best_only=False,
#                                        save_freq='epoch')
#
#     history = model.fit(images_train, masks_train,
#                         batch_size=batch_size, epochs=nb_epoch, verbose=1, shuffle=True,
#                         callbacks=[model_checkpoint, csv_logger])
#
# def check_gpu():
#     gpus = tf.config.list_physical_devices('GPU')
#     if gpus:
#         print("GPUs are available:")
#         for gpu in gpus:
#             print(f"  {gpu}")
#     else:
#         print("No GPUs found.")
#
# if __name__ == "__main__":
#     check_gpu()  # Check if GPU is available before starting training
#     #let's ignore the value from cur_fold, and train all the folds
#     for fold_nr in range(4):
#         train(str(fold_nr), plane, batch_size, epoch)
#         # train(cur_fold, plane, batch_size, epoch, init_lr)
#     print("Training done")
