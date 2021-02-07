import numpy as np
import os
import tensorflow as tf
import utils.helper as helper
import utils.loss_handler as loss

from keras.callbacks import ModelCheckpoint, LearningRateScheduler, Callback
from keras.initializers import Constant
from keras.layers import Dense, Lambda, BatchNormalization, Activation, Concatenate, GlobalAveragePooling2D
from keras.layers.convolutional import Conv2D
from keras.models import Model, Input
from keras.optimizers import Adam
from keras import regularizers
from keras import backend as K
#from keras.utils import plot_model


# Set parameters
lr = 0.0002
kr = 0.0
nb_epoch = 100
batch_size = 4
result_dir = 'results'
file_path_train = 'dataset_test.txt'
file_path_validation = 'dataset_test.txt'

# Get train and validation dataset
dataset_train = helper.get_dataset(file_path_train)
dataset_validation = helper.get_dataset(file_path_validation)

X_train = np.squeeze(np.array(dataset_train.images))
X_validation = np.squeeze(np.array(dataset_validation.images))
y_train = np.squeeze(np.array(dataset_train.poses))
y_validation = np.squeeze(np.array(dataset_validation.poses))

y_train_quaternion = y_train[:,:4]
y_train_translation = y_train[:,4:7]
y_validation_quaternion = y_validation[:,:4]
y_validation_translation = y_validation[:,4:7]

X_train = X_train.astype('float32')
X_validation = X_validation.astype('float32')

input_height, input_width, input_channel = X_train.shape[1], X_train.shape[2], X_train.shape[3]
input_shape = (input_height, input_width, input_channel)

# Set learning weights
lambda_epi = K.variable(0.6)
lambda_ssim = K.variable(1.0)
lambda_l1 = K.variable(0.3)


class MyCallback(Callback):

    def __init__(self, lambda_epi, lambda_ssim, lambda_l1):
      self.lambda_epi = lambda_epi
      self.lambda_ssim = lambda_ssim
      self.lambda_l1 = lambda_l1

    def on_epoch_end(self, epoch, log={}):
        if epoch in [2, 4, 6, 8, 10, 12, 14, 16, 18]:
            K.set_value(self.lambda_epi, K.get_value(self.lambda_epi) - 0.05)

def norm_clip(x):
        return tf.clip_by_norm(x, 1, axes=[1])

def lr_schedule(epoch):
    lr = 0.0002
    if epoch > 60:
        lr = 0.0001
    return lr


if __name__ == "__main__":

    # Pre-process input data
    input_tensor = Input(
        shape = input_shape,
        name='direct_epipolar')
    input_of = Lambda(
        lambda x: x[:,:,:,:2],
        output_shape = (input_height, input_width, 2))(input_tensor)
    frame_t0 = Lambda(
        lambda x: x[:,:,:,2:5],
        output_shape = (input_height, input_width, 3))(input_tensor)
    frame_t2 = Lambda(
        lambda x: x[:,:,:,5:8],
        output_shape = (input_height, input_width, 3))(input_tensor)

    # Stack up network blocks
    convraw1_1 = Conv2D(
        filters = 16,
        kernel_size = 7,
        strides = (2,2),
        padding = 'same',
        activation = None,
        kernel_regularizer = regularizers.l2(kr))(frame_t0)
    convraw_bn1_1 = BatchNormalization()(convraw1_1)
    convraw_relu1_1 = Activation('relu')(convraw_bn1_1)
    convraw1_2 = Conv2D(
        filters = 32,
        kernel_size = 5,
        strides = (2, 2),
        padding = 'same',
        activation = None,
        kernel_regularizer = regularizers.l2(kr))(convraw_relu1_1)
    convraw_bn1_2 = BatchNormalization()(convraw1_2)
    convraw_relu1_2 = Activation('relu')(convraw_bn1_2)

    convraw2_1 = Conv2D(
        filters = 16,
        kernel_size = 7,
        strides = (2, 2),
        padding = 'same',
        activation = None,
        kernel_regularizer = regularizers.l2(kr))(frame_t2)
    convraw_bn2_1 = BatchNormalization()(convraw2_1)
    convraw_relu2_1 = Activation('relu')(convraw_bn2_1)
    convraw2_2 = Conv2D(
        filters = 32,
        kernel_size = 5,
        strides = (2, 2),
        padding = 'same',
        activation = None,
        kernel_regularizer = regularizers.l2(kr))(convraw_relu2_1)
    convraw_bn2_2 = BatchNormalization()(convraw2_2)
    convraw_relu2_2 = Activation('relu')(convraw_bn2_2)

    concat1 = Concatenate()([convraw_relu1_2, convraw_relu2_2])

    convraw3_1 = Conv2D(
        filters = 64,
        kernel_size = 3,
        strides = (2, 2),
        padding = 'same',
        activation = None,
        kernel_regularizer = regularizers.l2(kr))(concat1)
    convraw_bn3_1 = BatchNormalization()(convraw3_1)
    convraw_relu3_1 = Activation('relu')(convraw_bn3_1)
    convraw3_2 = Conv2D(
        filters = 128,
        kernel_size = 3,
        strides = (2, 2),
        padding = 'same',
        activation = None,
        kernel_regularizer = regularizers.l2(kr))(convraw_relu3_1)
    convraw_bn3_2 = BatchNormalization()(convraw3_2)
    convraw_relu3_2 = Activation('relu')(convraw_bn3_2)
    convraw3_3 = Conv2D(
        filters = 256,
        kernel_size = 3,
        strides = (2, 2),
        padding = 'same',
        activation = None,
        kernel_regularizer = regularizers.l2(kr))(convraw_relu3_2)
    convraw_bn3_3 = BatchNormalization()(convraw3_3)
    convraw_relu3_3 = Activation('relu')(convraw_bn3_3)
    convraw3_4 = Conv2D(
        filters = 256,
        kernel_size = 3,
        strides = (2, 2),
        padding = 'same',
        activation = None,
        kernel_regularizer = regularizers.l2(kr))(convraw_relu3_3)
    convraw_bn3_4 = BatchNormalization()(convraw3_4)
    convraw_relu3_4 = Activation('relu')(convraw_bn3_4)
    avg_pool1 = GlobalAveragePooling2D()(convraw_relu3_4)

    conv1 = Conv2D(
        filters = 16,
        kernel_size = 7,
        strides = (2, 2),
        padding = 'same',
        activation = None,
        kernel_regularizer = regularizers.l2(kr))(input_of)
    conv1_bn = BatchNormalization()(conv1)
    conv1_relu = Activation('relu')(conv1_bn)
    conv2 = Conv2D(
        filters = 32,
        kernel_size = 5,
        strides = (2, 2),
        padding = 'same',
        activation = None,
        kernel_regularizer = regularizers.l2(kr))(conv1_relu)
    conv2_bn = BatchNormalization()(conv2)
    conv2_relu = Activation('relu')(conv2_bn)
    conv3 = Conv2D(
        filters = 64,
        kernel_size = 3,
        strides = (2, 2),
        padding = 'same',
        activation = None,
        kernel_regularizer = regularizers.l2(kr))(conv2_relu)
    conv3_bn = BatchNormalization()(conv3)
    conv3_relu = Activation('relu')(conv3_bn)
    conv4 = Conv2D(
        filters = 128,
        kernel_size = 3,
        strides = (2, 2),
        padding = 'same',
        activation = None,
        kernel_regularizer = regularizers.l2(kr))(conv3_relu)
    conv4_bn = BatchNormalization()(conv4)
    conv4_relu = Activation('relu')(conv4_bn)
    conv5 = Conv2D(
        filters = 256,
        kernel_size = 3,
        strides = (2, 2),
        padding = 'same',
        activation = None,
        kernel_regularizer = regularizers.l2(kr))(conv4_relu)
    conv5_bn = BatchNormalization()(conv5)
    conv5_relu = Activation('relu')(conv5_bn)
    conv6 = Conv2D(
        filters = 256,
        kernel_size = 3,
        strides = (2, 2),
        padding = 'same',
        activation = None,
        kernel_regularizer = regularizers.l2(kr))(conv5_relu)
    conv6_bn = BatchNormalization()(conv6)
    conv6_relu = Activation('relu')(conv6_bn)
    avg_pool2 = GlobalAveragePooling2D()(conv6_relu)

    output_q01 = Dense(
        units = 4,
        activation = norm_clip,
        kernel_initializer = 'zero',
        bias_initializer = Constant(value=[1.0, 0.0, 0.0, 0.0]),
        kernel_regularizer = regularizers.l2(kr))(avg_pool2)
    output_t01 = Dense(
        units = 3,
        activation = norm_clip,
        kernel_regularizer = regularizers.l2(kr))(avg_pool2)
    output_q02 = Dense(
        units = 4,
        activation = norm_clip,
        kernel_initializer = 'zero',
        bias_initializer = Constant(value=[1.0, 0.0, 0.0, 0.0]),
        kernel_regularizer = regularizers.l2(kr))(avg_pool1)
    output_t02 = Dense(
        units = 3,
        activation = None,
        name = 'tran_2',
        kernel_regularizer = regularizers.l2(kr))(avg_pool1)

    output_q01 = Lambda(
        lambda x: K.l2_normalize((x + K.epsilon()), axis=1),
        name = 'quat_1')(output_q01)
    output_t01 = Lambda(
        lambda x: K.l2_normalize((x + K.epsilon()), axis=1),
        name = 'tran_1')(output_t01)
    output_q02 = Lambda(
        lambda x: K.l2_normalize((x + K.epsilon()), axis=1),
        name = 'quat_2')(output_q02)

    model = Model(
        inputs = [input_tensor],
        outputs = [output_q01, output_t01, output_q02, output_t02])
    #plot_model(model, to_file='results/model.png')
    model.summary()

    # Set optimizer
    optimizer = Adam(
        lr = lr,
        beta_1 = 0.9,
        beta_2 = 0.999,
        epsilon = None,
        decay = 0.0,
        amsgrad = False)

    # Compile model
    model.compile(
        optimizer = optimizer,
        loss = [
            loss.get_dummy_loss,
            loss.get_epipolar_loss(input_of, output_q01),
            loss.get_ssim_loss(input_of, frame_t0, frame_t2, output_q01, output_t01, output_t02),
            loss.get_l1_loss(input_of, frame_t0, frame_t2, output_q01, output_q02, output_t01)],
        loss_weights = [0.0, lambda_epi, lambda_ssim, lambda_l1])

    # Setup checkpointer
    checkpointer = ModelCheckpoint(
        filepath = os.path.join(result_dir, "checkpoint_weights.h5"),
        verbose = 1,
        save_best_only = True,
        save_weights_only = True)

    # Train the model
    history = model.fit(
        [X_train],
        [y_train_quaternion, y_train_translation, y_train_quaternion, y_train_translation],
        batch_size = batch_size,
        epochs = nb_epoch,
        validation_data = [
            [X_validation],
            [y_validation_quaternion,
             y_validation_translation,
             y_validation_quaternion,
             y_validation_translation]],
        callbacks = [
            checkpointer,
            LearningRateScheduler(lr_schedule),
            MyCallback(lambda_epi, lambda_ssim, lambda_l1)],
        shuffle = True,
        verbose = 1)

    # Save weights
    model.save_weights(os.path.join(result_dir, 'CNN_weights.h5'))

    # Plot and save history
    helper.plot_history(history)
    helper.save_history(history, os.path.join(result_dir, 'history_train.txt'))