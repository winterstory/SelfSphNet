import math
import numpy as np
import os
import tensorflow as tf
import time
import utils.helper as helper

from keras import backend as K
from keras import regularizers
from keras.layers.convolutional import Conv2D
from keras.layers import Dense, Lambda, BatchNormalization, Activation, Concatenate, GlobalAveragePooling2D
from keras.models import Model, Input
#from keras.utils import plot_model


# Set parameters
result_dir = 'results'
file_path_test = 'dataset_test.txt'
kr = 0.0

# Get test dataset
dataset_test = helper.get_dataset(file_path_test)

X_test = np.squeeze(np.array(dataset_test.images))
y_test = np.squeeze(np.array(dataset_test.poses))
X_test = X_test.astype('float32')

y_test_quaternion = y_test[:,:4]
y_test_translation = y_test[:,4:7]

img_height, img_width, img_channel = X_test.shape[1], X_test.shape[2], X_test.shape[3]
input_shape = (img_height, img_width, img_channel)


def norm_clip(x):
    return tf.clip_by_norm(x, 1, axes=[1])


if __name__ == "__main__":

  # Starting point
  start = time.time()

  input_tensor = Input(
    shape = input_shape,
    name = 'direct_epipolar')
  input_of = Lambda(
    lambda x: x[:,:,:,:2],
    output_shape = (img_height, img_width, 2))(input_tensor)
  frame_t0 = Lambda(
    lambda x: x[:,:,:,2:5],
    output_shape = (img_height, img_width, 3))(input_tensor)
  frame_t2 = Lambda(
    lambda x: x[:,:,:,5:8],
    output_shape = (img_height, img_width, 3))(input_tensor)

  # Network structure
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
    strides = (2,2),
    padding = 'same',
    activation = None,
    kernel_regularizer = regularizers.l2(kr))(convraw_relu1_1)
  convraw_bn1_2 = BatchNormalization()(convraw1_2)
  convraw_relu1_2 = Activation('relu')(convraw_bn1_2)

  convraw2_1 = Conv2D(
    filters = 16,
    kernel_size = 7,
    strides = (2,2),
    padding = 'same',
    activation = None,
    kernel_regularizer = regularizers.l2(kr))(frame_t2)
  convraw_bn2_1 = BatchNormalization()(convraw2_1)
  convraw_relu2_1 = Activation('relu')(convraw_bn2_1)
  convraw2_2 = Conv2D(
    filters = 32,
    kernel_size = 5,
    strides = (2,2),
    padding='same',
    activation = None,
    kernel_regularizer = regularizers.l2(kr))(convraw_relu2_1)
  convraw_bn2_2 = BatchNormalization()(convraw2_2)
  convraw_relu2_2 = Activation('relu')(convraw_bn2_2)

  concat1 = Concatenate()([convraw_relu1_2, convraw_relu2_2])

  convraw3_1 = Conv2D(
    filters = 64,
    kernel_size = 3,
    strides = (2,2),
    padding = 'same',
    activation = None,
    kernel_regularizer = regularizers.l2(kr))(concat1)
  convraw_bn3_1 = BatchNormalization()(convraw3_1)
  convraw_relu3_1 = Activation('relu')(convraw_bn3_1)
  convraw3_2 = Conv2D(
    filters= 128,
    kernel_size = 3,
    strides = (2,2),
    padding = 'same',
    activation = None,
    kernel_regularizer = regularizers.l2(kr))(convraw_relu3_1)
  convraw_bn3_2 = BatchNormalization()(convraw3_2)
  convraw_relu3_2 = Activation('relu')(convraw_bn3_2)
  convraw3_3 = Conv2D(
    filters = 256,
    kernel_size = 3,
    strides = (2,2),
    padding = 'same',
    activation = None,
    kernel_regularizer = regularizers.l2(kr))(convraw_relu3_2)
  convraw_bn3_3 = BatchNormalization()(convraw3_3)
  convraw_relu3_3 = Activation('relu')(convraw_bn3_3)
  convraw3_4 = Conv2D(
    filters = 256,
    kernel_size = 3,
    strides = (2,2),
    padding = 'same',
    activation = None,
    kernel_regularizer = regularizers.l2(kr))(convraw_relu3_3)
  convraw_bn3_4 = BatchNormalization()(convraw3_4)
  convraw_relu3_4 = Activation('relu')(convraw_bn3_4)
  avg_pool1 = GlobalAveragePooling2D()(convraw_relu3_4)

  conv1 = Conv2D(
    filters = 16,
    kernel_size = 7,
    strides = (2,2),
    padding = 'same',
    activation = None,
    kernel_regularizer = regularizers.l2(kr))(input_of)
  conv1_bn = BatchNormalization()(conv1)
  conv1_relu = Activation('relu')(conv1_bn)
  conv2 = Conv2D(
    filters = 32,
    kernel_size = 5,
    strides = (2,2),
    padding = 'same',
    activation = None,
    kernel_regularizer = regularizers.l2(kr))(conv1_relu)
  conv2_bn = BatchNormalization()(conv2)
  conv2_relu = Activation('relu')(conv2_bn)
  conv3 = Conv2D(
    filters = 64,
    kernel_size = 3,
    strides = (2,2),
    padding = 'same',
    activation = None,
    kernel_regularizer = regularizers.l2(kr))(conv2_relu)
  conv3_bn = BatchNormalization()(conv3)
  conv3_relu = Activation('relu')(conv3_bn)
  conv4 = Conv2D(
    filters = 128,
    kernel_size = 3,
    strides = (2,2),
    padding = 'same',
    activation = None,
    kernel_regularizer = regularizers.l2(kr))(conv3_relu)
  conv4_bn = BatchNormalization()(conv4)
  conv4_relu = Activation('relu')(conv4_bn)
  conv5 = Conv2D(
    filters = 256,
    kernel_size = 3,
    strides = (2,2),
    padding = 'same',
    activation = None,
    kernel_regularizer = regularizers.l2(kr))(conv4_relu)
  conv5_bn = BatchNormalization()(conv5)
  conv5_relu = Activation('relu')(conv5_bn)
  conv6 = Conv2D(
    filters = 256,
    kernel_size = 3,
    strides = (2,2),
    padding = 'same',
    activation = None,
    kernel_regularizer = regularizers.l2(kr))(conv5_relu)
  conv6_bn = BatchNormalization()(conv6)
  conv6_relu = Activation('relu')(conv6_bn)
  avg_pool2 = GlobalAveragePooling2D()(conv6_relu)

  output_q01 = Dense(
    units = 4,
    activation = None,
    name = 'out_quaternion_1')(avg_pool2)
  output_t01 = Dense(
    units = 3,
    activation = None,
    name='out_translation_1')(avg_pool2)
  output_q02 = Dense(
    units = 4,
    activation = None,
    name = 'out_quaternion_2')(avg_pool1)
  output_t02 = Dense(
    units = 3,
    activation = None,
    name='out_translation_2')(avg_pool1)

  output_q01 = Lambda(lambda x: K.l2_normalize((x + K.epsilon()), axis = 1))(output_q01)
  output_t01 = Lambda(lambda x: K.l2_normalize((x + K.epsilon()), axis = 1))(output_t01)
  output_q02 = Lambda(lambda x: K.l2_normalize((x + K.epsilon()), axis = 1))(output_q02)

  model = Model(
    inputs = [input_tensor],
    outputs = [output_q01, output_t01, output_q02, output_t02])
  #plot_model(model, to_file='results/model.png', show_shapes=True)
  model.summary()

  # Load tested weights
  model.load_weights(os.path.join(result_dir, 'checkpoint_weights.h5'))
  print("weights.h5 Loaded")

  out_quaternion_1, out_translation_1, out_quaternion_2, out_translation_2 = model.predict(X_test)

  rot_23 = np.zeros((len(dataset_test.images), 2))
  tran_23 = np.zeros((len(dataset_test.images), 2))
  rot_12 = np.zeros((len(dataset_test.images), 2))
  tran_12 = np.zeros((len(dataset_test.images), 2))

  np.set_printoptions(precision=5)
  gt_quat_init = np.array([
    y_test_quaternion[0][0],
    y_test_quaternion[0][1],
    y_test_quaternion[0][2],
    y_test_quaternion[0][3]])
  er_quat_init = np.array([
    out_quaternion_1[0][0],
    out_quaternion_1[0][1],
    out_quaternion_1[0][2],
    out_quaternion_1[0][3]])
  gt_tran_init = np.array([
    y_test_translation[0][0],
    y_test_translation[0][1],
    y_test_translation[0][2]])
  er_tran_init = np.array([
    out_translation_1[0][0],
    out_translation_1[0][1],
    out_translation_1[0][2]])

  print(' ')
  with open('results/results.txt', 'w') as file:
    file.write('[Data number, gt_quat, er_quat, gt_tran, er_tran]\n')
    print('  '+'Ground Truth Rotation'+'                 '+'Estimated Rotation'+'                    '+
          'Ground Truth Translation'+'     '+'Estimated Translation'+'        '+'Rotation'+'    '+'Translation')
    for i in range(len(dataset_test.images)-1):
      gt_quaternion_12 = y_test_quaternion[i]
      gt_translation_12 = y_test_translation[i]
      gt_quaternion_23 = y_test_quaternion[i + 1]
      gt_translation_23 = y_test_translation[i + 1]

      er_quaternion_12 = out_quaternion_1[i]
      er_translation_12 = out_translation_1[i]

      out_euler_12 = helper.convert_quaternion_to_euler(
        out_quaternion_1[i][0],
        out_quaternion_1[i][1],
        out_quaternion_1[i][2],
        out_quaternion_1[i][3])
      out_euler_13 = helper.convert_quaternion_to_euler(
        out_quaternion_2[i][0],
        out_quaternion_2[i][1],
        out_quaternion_2[i][2],
        out_quaternion_2[i][3])
      out_euler_23 = out_euler_13 - out_euler_12

      er_quaternion_23 = helper.convert_euler_to_quaternion(
        out_euler_23[0],
        out_euler_23[1],
        out_euler_23[2])

      gt_translation_23 = gt_translation_23
      er_translation_23 = (out_translation_2[i] - er_translation_12) * np.linalg.norm(gt_translation_12)

      # Error evaluation for photometric reprojection loss
      d_q_23 = abs(np.sum(np.multiply(gt_quaternion_23, er_quaternion_23)))
      error_rot_23 = np.round(2 * np.arccos(d_q_23) * 180 / math.pi, 5)
      error_tran_23 = np.round(np.linalg.norm(gt_translation_23 - er_translation_23), 5)
      rot_23[i, :] = [error_rot_23]
      tran_23[i, :] = [error_tran_23]
      print(
        i + 1,
        gt_quaternion_23,
        er_quaternion_23,
        gt_translation_23,
        er_translation_23,
        str(error_rot_23) + '[deg]',
        str(error_tran_23) + '[m]')

      # Error evaluation for epipolar angular loss
      d_q_12 = abs(np.sum(np.multiply(gt_quaternion_12, er_quaternion_12)))
      error_rot_12 = np.round(2 * np.arccos(d_q_12) * 180 / math.pi, 5)
      gt_translation_12 /= np.linalg.norm(gt_translation_12)
      er_translation_12 /= np.linalg.norm(er_translation_12)
      d_t_12 = abs(np.sum(np.multiply(gt_translation_12, er_translation_12)))
      error_tran_12 = np.round(np.arccos(d_t_12) * 180 / math.pi, 5)
      rot_12[i, :] = [error_rot_12]
      tran_12[i, :] = [error_tran_12]
      # print(
      #   i + 1,
      #   gt_quaternion_12,
      #   er_quaternion_12,
      #   gt_translation_12,
      #   er_translation_12,
      #   str(error_rot_12) + '[deg]',
      #   str(error_tran_12) + '[deg]')

  # Handle nan value to zero
  rot_12[np.isnan(rot_12)] = 0
  rot_23[np.isnan(rot_23)] = 0
  tran_12[np.isnan(tran_12)] = 0
  tran_23[np.isnan(tran_23)] = 0

  med_rot_23 = np.median(rot_23, axis=0)
  avg_rot_23 = np.average(rot_23, axis=0)
  std_rot_23 = np.std(rot_23, axis=0)
  med_tran_23 = np.median(tran_23, axis=0)
  avg_tran_23 = np.average(tran_23, axis=0)
  std_tran_23 = np.std(tran_23, axis=0)

  med_rot_12 = np.median(rot_12, axis=0)
  avg_rot_12 = np.average(rot_12, axis=0)
  std_rot_12 = np.std(rot_12, axis=0)
  med_tran_12 = np.median(tran_12, axis=0)
  avg_tran_12 = np.average(tran_12, axis=0)
  std_tran_12 = np.std(tran_12, axis=0)

  print(' ')
  print(
    'Median rot_23. error ',
    np.round(med_rot_23[1], 5),
    '[deg].')
  print(
    'Average rot_23. error ',
    np.round(avg_rot_23[1], 5),
    '+/-',
    np.round(std_rot_23[1], 5),
    '[deg].')
  print(
    'Median tran_23. error ',
    np.round(med_tran_23[1], 5),
    '[m].')
  print(
    'Average tran_23. error ',
    np.round(avg_tran_23[1], 5),
    '+/-',
    np.round(std_tran_23[1], 5),
    '[m].')
  print(' ')
  print(
    'Median rot_12. error ',
    np.round(med_rot_12[1], 5),
    '[deg].')
  print(
    'Average rot_12. error ',
    np.round(avg_rot_12[1], 5),
    '+/-',
    np.round(std_rot_12[1], 5),
    '[deg].')
  print(
    'Median tran_12. error ',
    np.round(med_tran_12[1], 5),
    '[m].')
  print(
    'Average tran_12. error ',
    np.round(avg_tran_12[1], 5),
    '+/-',
    np.round(std_tran_12[1], 5),
    '[deg].')
