import cv2
import matplotlib.pyplot as plt
import math
import numpy as np
import os
import tensorflow as tf

from tqdm import tqdm
from keras import backend as K


class datasource(object):
    def __init__(self, images, poses):
        self.images = images
        self.poses = poses


def preprocess_image(images_1, images_2, images_3):
    images_out = []
    for i in tqdm(list(range(len(images_1)))):
        optical_flow = np.load(images_1[i])
        frame_1 = cv2.imread(images_2[i])
        frame_1 = cv2.cvtColor(frame_1, cv2.COLOR_BGR2RGB)
        frame_2 = cv2.imread(images_3[i])
        frame_2 = cv2.cvtColor(frame_2, cv2.COLOR_BGR2RGB)

        frame_1 = frame_1 / 255.0
        frame_2 = frame_2 / 255.0
        frame_1 = frame_1.astype('float32')
        frame_2 = frame_2.astype('float32')

        X = np.dstack([optical_flow, frame_1, frame_2])
        images_out.append(X)
    return images_out


def get_data(dataset):
    directory = 'datasets/'
    poses = []
    images_1 = []
    images_2 = []
    images_3 = []
    with open(directory + dataset) as f:
        next(f)
        next(f)
        next(f)
        for line in f:
            fname_1,fname_2,fname_3,p0,p1,p2,p3,p4,p5,p6 = line.split()
            p0 = float(p0)
            p1 = float(p1)
            p2 = float(p2)
            p3 = float(p3)
            p4 = float(p4)
            p5 = float(p5)
            p6 = float(p6)
            poses.append((p0,p1,p2,p3,p4,p5,p6))
            images_1.append(directory + fname_1)
            images_2.append(directory + fname_2)
            images_3.append(directory + fname_3)
    images = preprocess_image(images_1, images_2, images_3)
    return datasource(images, poses)


def get_dataset(file_path):
    dataset = get_data(file_path)

    images = []
    poses = []

    for i in range(len(dataset.images)):
        images.append(dataset.images[i])
        poses.append(dataset.poses[i])
    return datasource(images, poses)


def convert_quaternion_to_euler(qw, qx, qy, qz):
    roll = np.arctan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx ** 2 + qy ** 2))
    pitch = np.arcsin(2.0 * (qw * qy - qz * qx))
    yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy ** 2 + qz ** 2))
    return np.array([roll, pitch, yaw])


def convert_euler_to_quaternion(roll, pitch, yaw):
    sr = np.sin(roll / 2.0)
    cr = np.cos(roll / 2.0)
    sp = np.sin(pitch / 2.0)
    cp = np.cos(pitch / 2.0)
    sy = np.sin(yaw / 2.0)
    cy = np.cos(yaw / 2.0)

    # ZYX rotation
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return np.array([qw, qx, qy, qz])


def plot_history(history):
    result_dir = 'results'
    plt.plot(history.history['loss'], color='green')
    plt.plot(history.history['tran_1_loss'], color='green', ls=":")
    plt.plot(history.history['quat_2_loss'], color='green', ls="--")

    plt.plot(history.history['val_loss'], color='orange')
    plt.plot(history.history['val_tran_1_loss'], color='orange', ls=":")
    plt.plot(history.history['val_quat_2_loss'], color='orange', ls="--")

    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.ylim([0, 0.6])
    plt.legend(['loss', 'val_loss'], loc='upper right')
    plt.savefig(os.path.join(result_dir, 'loss.png'))


def save_history(history, result_file):
    loss = history.history['loss']
    tran_1_loss = history.history['tran_1_loss']
    quat_2_loss = history.history['quat_2_loss']
    tran_2_loss = history.history['tran_2_loss']
    val_loss = history.history['val_loss']
    val_tran_1_loss = history.history['val_tran_1_loss']
    val_quat_2_loss = history.history['val_quat_2_loss']
    val_tran_2_loss = history.history['val_tran_2_loss']
    nb_epoch = len(loss)

    with open(result_file, "w") as fp:
        fp.write("epoch\tloss\t\tepi_loss\tSSIM_loss\tL1_loss\t\t"
                 "val_loss\tval_epi_loss\tval_SSIM_loss\tval_L1_loss\n")
        for i in range(nb_epoch):
            fp.write("%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n" %
                     (i, loss[i], tran_1_loss[i], quat_2_loss[i], tran_2_loss[i],
                      val_loss[i], val_tran_1_loss[i], val_quat_2_loss[i], val_tran_2_loss[i]))


def supervision(y_true, y_pred):
    loss = K.sqrt(K.sum(K.square(y_true[:, :] - y_pred[:, :]), axis=1, keepdims=True))
    return loss


def read_npy(input_path, height, width, dim):
    input_data = np.load(input_path)
    input_data = tf.convert_to_tensor(input_data, dtype=tf.float32)
    input_data = tf.reshape(input_data, [height, width, dim])
    return tf.expand_dims(input_data, 0)


def read_image(image_path, shape):
    if image_path.lower().endswith("png"):
        image = tf.image.decode_png(tf.io.read_file(image_path))
    else:
        image = tf.image.decode_jpeg(tf.io.read_file(image_path))
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, shape, tf.image.ResizeMethod.AREA)
    return tf.expand_dims(image, 0)


def get_tf_batch_matrix(matrix):
    return tf.transpose(tf.stack(matrix), [2, 0, 1])


def apply_gauss_filter(fx, sigma):
    x = tf.range(-int(fx / 2), int(fx / 2) + 1, 1)
    y = x
    Y, X = tf.meshgrid(x, y)

    sigma = -2 * (sigma ** 2)
    z = tf.cast(tf.add(tf.square(X), tf.square(Y)), tf.float32)
    k = 2 * tf.exp(tf.divide(z, sigma))
    k = tf.divide(k, tf.reduce_sum(k))
    return k


def apply_gaussian_blur(image, filtersize, sigma, n_channels):
    fx = filtersize[0]
    fil = apply_gauss_filter(fx, sigma)
    fil = tf.stack([fil] * n_channels, axis=2)
    fil = tf.expand_dims(fil, 3)

    new = tf.image.convert_image_dtype(image, tf.dtypes.float32)
    new = tf.tile(new, [1, 1, 1, 1])
    gaussian_blur = tf.nn.depthwise_conv2d(
        new,
        fil,
        strides = [1, 1, 1, 1],
        padding = "SAME")
    return gaussian_blur


def apply_median_blur(x, filter_size, strides = None):
    if strides is None:
        strides = [1, 1, 1, 1]
    patches = tf.extract_image_patches(
        x,
        [1, filter_size, filter_size, 1],
        strides,
        4 * [1],
        'SAME',
        name = "median_pool")
    patches = tf.expand_dims(patches, -1)
    median_blur = tf.contrib.distributions.percentile(
        patches,
        50,
        axis = 3,
        keep_dims = False)
    return median_blur


def get_3d_optical_flow(width, height, optical_flow):
    tf_batch_size = tf.shape(optical_flow)[0]
    tf_height = tf.shape(optical_flow)[1]
    tf_width = tf.shape(optical_flow)[2]

    # u,v tensor
    u_tensor = K.tile(K.arange(0, width, step = 1, dtype = 'float32'), [height])
    u_tensor = K.reshape(u_tensor, [height, width])

    v_tensor = K.tile(K.arange(0, height, step = 1, dtype = 'float32'), [width])
    v_tensor = K.transpose(K.reshape(v_tensor, [width, height]))

    # 3D location original
    X = K.sin(math.pi * u_tensor / height) * K.sin(math.pi * v_tensor / height)
    Y = K.cos(math.pi * u_tensor / height) * K.sin(math.pi * v_tensor / height)
    Z = K.cos(math.pi * v_tensor / height)
    XYZ = K.tile(
        K.expand_dims(
            K.reshape(K.stack([X, Y, Z]), [3, tf_height * tf_width]), 0), [tf_batch_size, 1, 1])

    # Optical flow, FX/FY/FZ original
    FX = K.sin(math.pi * (u_tensor + optical_flow[:, :, :, 0]) / height) * \
         K.sin(math.pi * (v_tensor + optical_flow[:, :, :, 1]) / height)
    FY = K.cos(math.pi * (u_tensor + optical_flow[:, :, :, 0]) / height) * \
         K.sin(math.pi * (v_tensor + optical_flow[:, :, :, 1]) / height)
    FZ = K.cos(math.pi * (v_tensor + optical_flow[:, :, :, 1]) / height)
    return FX, FY, FZ, XYZ, Z