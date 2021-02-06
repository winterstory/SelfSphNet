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


# Get test data
dataset_test = helper.get_test_data()

X_test = np.squeeze(np.array(dataset_test.images))
y_test = np.squeeze(np.array(dataset_test.poses))
X_test = X_test.astype('float32')

y_test_q = y_test[:,:4]
y_test_t = y_test[:,4:7]

# Set parameters
result_dir = 'results'
img_height, img_width, img_channel = X_test.shape[1], X_test.shape[2], X_test.shape[3]
input_shape = (img_height, img_width, img_channel)
kr = 0.0

def norm_clip(x):
  return tf.clip_by_norm(x, 1, axes=[1])

def quaternion_to_euler(w, x, y, z):
  roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x ** 2 + y ** 2))
  pitch = np.arcsin(2.0 * (w * y - z * x))
  yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y ** 2 + z ** 2))

  return np.array([roll, pitch, yaw])

def euler_to_quaternion(roll, pitch, yaw):
  sr = np.sin(roll / 2.0)
  cr = np.cos(roll / 2.0)
  sp = np.sin(pitch / 2.0)
  cp = np.cos(pitch / 2.0)
  sy = np.sin(yaw / 2.0)
  cy = np.cos(yaw / 2.0)

  # ZYX rotation
  w = cr * cp * cy + sr * sp * sy
  x = sr * cp * cy - cr * sp * sy
  y = cr * sp * cy + sr * cp * sy
  z = cr * cp * sy - sr * sp * cy

  return np.array([w, x, y, z])

def tran_error(gt, er):
  q1 = gt / np.linalg.norm(gt)
  q2 = er / np.linalg.norm(er)
  d_q = abs(np.sum(np.multiply(q1, q2)))
  tran = np.arccos(d_q) * 180 / math.pi

  return tran

def rot_error(gt, er):
  q1 = gt / np.linalg.norm(gt)
  q2 = er / np.linalg.norm(er)
  d_q = abs(np.sum(np.multiply(q1, q2)))
  rot = 2 * np.arccos(d_q) * 180 / math.pi
  return rot


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
    name = 'quat_1')(avg_pool2)
  output_t01 = Dense(
    units = 3,
    activation = None,
    name='tran_1')(avg_pool2)
  output_q02 = Dense(
    units = 4,
    activation = None,
    name = 'quat_2')(avg_pool1)
  output_t02 = Dense(
    units = 3,
    activation = None,
    name='tran_2')(avg_pool1)

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

  quat_1, tran_1, quat_2, tran_2 = model.predict(X_test)

  rot_23 = np.zeros((len(dataset_test.images), 2))
  tran_23 = np.zeros((len(dataset_test.images), 2))
  rot_12 = np.zeros((len(dataset_test.images), 2))
  tran_12 = np.zeros((len(dataset_test.images), 2))

  np.set_printoptions(precision=5)
  gt_quat_init = np.array([y_test_q[0][0], y_test_q[0][1], y_test_q[0][2], y_test_q[0][3]])
  er_quat_init = np.array([quat_1[0][0], quat_1[0][1], quat_1[0][2], quat_1[0][3]])
  gt_tran_init = np.array([y_test_t[0][0], y_test_t[0][1], y_test_t[0][2]])
  er_tran_init = np.array([tran_1[0][0], tran_1[0][1], tran_1[0][2]])

  print(' ')
  with open('results/results.txt', 'w') as file:
    file.write('[Data number, gt_quat, er_quat, gt_tran, er_tran]\n')
    #print(gt_quat_init, er_quat_init, gt_tran_init, er_tran_init)
    print('  '+'Ground Truth Rotation'+'                 '+'Estimated Rotation'+'                    '+
          'Ground Truth Translation'+'     '+'Estimated Translation'+'        '+'Rotation'+'    '+'Translation')
    for i in range(len(dataset_test.images)-1):
      gt_quat_12 = y_test_q[i]    # I1 -> I2
      gt_tran_12 = y_test_t[i]    # I1 -> I2
      gt_quat_23 = y_test_q[i+1]  # I2 -> I3
      gt_tran_23 = y_test_t[i+1]  # I2 -> I3

      er_quat_12 = quat_1[i]      # I1 -> I2
      er_tran_12 = tran_1[i]      # I1 -> I2

      euler_12 = quaternion_to_euler(quat_1[i][0], quat_1[i][1], quat_1[i][2], quat_1[i][3])
      euler_13 = quaternion_to_euler(quat_2[i][0], quat_2[i][1], quat_2[i][2], quat_2[i][3])
      euler_23 = euler_13 - euler_12

      er_quat_23 = euler_to_quaternion(euler_23[0], euler_23[1], euler_23[2])

      gt_tran_23 = gt_tran_23
      er_tran_23 = (tran_2[i] - er_tran_12) * np.linalg.norm(gt_tran_12)

      # Error evaluation for photometric reprojection loss
      d_q_23 = abs(np.sum(np.multiply(gt_quat_23, er_quat_23)))
      error_rot_23 = np.round(2 * np.arccos(d_q_23) * 180 / math.pi, 5)
      error_tran_23 = np.round(np.linalg.norm(gt_tran_23 - er_tran_23), 5)
      rot_23[i, :] = [error_rot_23]
      tran_23[i, :] = [error_tran_23]
      print(i+1, gt_quat_23, er_quat_23, gt_tran_23, er_tran_23, str(error_rot_23)+'[deg]', str(error_tran_23)+'[m]')

      # Error evaluation for epipolar angular loss
      d_q_12 = abs(np.sum(np.multiply(gt_quat_12, er_quat_12)))
      error_rot_12 = np.round(2 * np.arccos(d_q_12) * 180 / math.pi, 5)
      gt_tran_12 = gt_tran_12 / np.linalg.norm(gt_tran_12)
      er_tran_12 = er_tran_12 / np.linalg.norm(er_tran_12)
      d_t_12 = abs(np.sum(np.multiply(gt_tran_12, er_tran_12)))
      error_tran_12 = np.round(np.arccos(d_t_12) * 180 / math.pi, 5)
      rot_12[i, :] = [error_rot_12]
      tran_12[i, :] = [error_tran_12]
      #print(i+1, gt_quat_12, er_quat_12, gt_tran_12, er_tran_12, str(error_rot_12)+'[deg]', str(error_tran_12)+'[deg]')

  rot_12[np.isnan(rot_12)] = 0  # nan=0
  rot_23[np.isnan(rot_23)] = 0  # nan=0
  tran_12[np.isnan(tran_12)] = 0  # nan=0
  tran_23[np.isnan(tran_23)] = 0  # nan=0

  med_rot_23 = np.median(rot_23, axis=0)
  avg_rot_23, std_rot_23 = np.average(rot_23, axis=0), np.std(rot_23, axis=0)
  med_tran_23 = np.median(tran_23, axis=0)
  avg_tran_23, std_tran_23 = np.average(tran_23, axis=0), np.std(tran_23, axis=0)

  med_rot_12 = np.median(rot_12, axis=0)
  avg_rot_12, std_rot_12 = np.average(rot_12, axis=0), np.std(rot_12, axis=0)
  med_tran_12 = np.median(tran_12, axis=0)
  avg_tran_12, std_tran_12 = np.average(tran_12, axis=0), np.std(tran_12, axis=0)

  print(' ')
  print('Median rot_23. error ', np.round(med_rot_23[1], 5), '[deg].')
  print('Average rot_23. error ', np.round(avg_rot_23[1], 5), '+/-', np.round(std_rot_23[1], 5), '[deg].')
  print('Median tran_23. error ', np.round(med_tran_23[1], 5), '[m].')
  print('Average tran_23. error ', np.round(avg_tran_23[1], 5), '+/-', np.round(std_tran_23[1], 5), '[m].')
  print(' ')
  print('Median rot_12. error ', np.round(med_rot_12[1], 5), '[deg].')
  print('Average rot_12. error ', np.round(avg_rot_12[1], 5), '+/-', np.round(std_rot_12[1], 5), '[deg].')
  print('Median tran_12. error ', np.round(med_tran_12[1], 5), '[m].')
  print('Average tran_12. error ', np.round(avg_tran_12[1], 5), '+/-', np.round(std_tran_12[1], 5), '[deg].')
