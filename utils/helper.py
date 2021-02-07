import cv2
import matplotlib.pyplot as plt
import math
import numpy as np
import os
import tensorflow as tf
import utils.reconstruction as reconstruction

from tqdm import tqdm
from keras import backend as K


# Set directories
directory = 'datasets/'
dataset_train = 'dataset_train.txt'
dataset_validation = 'dataset_validation.txt'
dataset_test = 'dataset_test.txt'
result_dir = 'results'

# Set constant values
w, h = 200, 100
R = h / math.pi
rad = math.radians
cos, sin = math.cos, math.sin
eps = 0.00000001


class datasource(object):
	def __init__(self, images, poses):
		self.images = images
		self.poses = poses

def preprocess(images_1, images_2, images_3):
	images_out = []
	for i in tqdm(list(range(len(images_1)))):
		X_1 = np.load(images_1[i])  # get optical flow
		X_2 = cv2.imread(images_2[i])  # get rgb pre-frames
		X_2 = cv2.cvtColor(X_2, cv2.COLOR_BGR2RGB)
		X_3 = cv2.imread(images_3[i])  # get rgb post-frames
		X_3 = cv2.cvtColor(X_3, cv2.COLOR_BGR2RGB)

		X_2 = X_2 / 255.0
		X_3 = X_3 / 255.0
		X_2 = X_2.astype('float32')
		X_3 = X_3.astype('float32')

		X = np.dstack([X_1, X_2, X_3])
		images_out.append(X)

	return images_out

def get_data(dataset):
	poses = []
	images_1 = []
	images_2 = []
	images_3 = []
	with open(directory+dataset) as f:
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
			images_1.append(directory+fname_1)
			images_2.append(directory+fname_2)
			images_3.append(directory+fname_3)
	images = preprocess(images_1, images_2, images_3)

	return datasource(images, poses)

def get_train_data():
	datasource_train = get_data(dataset_train)

	images_train = []
	poses_train = []

	for i in range(len(datasource_train.images)):
		images_train.append(datasource_train.images[i])
		poses_train.append(datasource_train.poses[i])

	return datasource(images_train, poses_train)

def get_validation_data():
	datasource_validation = get_data(dataset_validation)

	images_validation = []
	poses_validation = []

	for i in range(len(datasource_validation.images)):
		images_validation.append(datasource_validation.images[i])
		poses_validation.append(datasource_validation.poses[i])

	return datasource(images_validation, poses_validation)

def get_test_data():
	datasource_test = get_data(dataset_test)

	images_test = []
	poses_test = []

	for i in range(len(datasource_test.images)):
		images_test.append(datasource_test.images[i])
		poses_test.append(datasource_test.poses[i])

	return datasource(images_test, poses_test)

def plot_history(history):
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
		fp.write("epoch\tloss\t\tepi_loss\tSSIM_loss\tL1_loss\t\tval_loss\tval_epi_loss\tval_SSIM_loss\tval_L1_loss\n")
		for i in range(nb_epoch):
			fp.write("%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n" % (i, loss[i], tran_1_loss[i], quat_2_loss[i], tran_2_loss[i],
				val_loss[i], val_tran_1_loss[i], val_quat_2_loss[i], val_tran_2_loss[i]))

def tf_batch_matrix(matrix):
	return tf.transpose(tf.stack(matrix), [2, 0, 1])

def gaussFilter(fx, fy, sigma):
	x = tf.range(-int(fx / 2), int(fx / 2) + 1, 1)
	y = x
	Y, X = tf.meshgrid(x, y)

	sigma = -2 * (sigma ** 2)
	z = tf.cast(tf.add(tf.square(X), tf.square(Y)), tf.float32)
	k = 2 * tf.exp(tf.divide(z, sigma))
	k = tf.divide(k, tf.reduce_sum(k))

	return k

def gaussian_blur(image, filtersize, sigma, n_channels):
	fx, fy = filtersize[0], filtersize[1]
	fil = gaussFilter(fx, fy, sigma)
	fil = tf.stack([fil] * n_channels, axis=2)
	fil = tf.expand_dims(fil, 3)

	new = tf.image.convert_image_dtype(image, tf.dtypes.float32)
	new = tf.tile(new, [1, 1, 1, 1])
	res = tf.nn.depthwise_conv2d(new, fil, strides=[1, 1, 1, 1], padding="SAME")

	return res

def median_blur(x, filter_size, strides=[1,1,1,1]):
	x_size = x.get_shape().as_list()

	patches = tf.extract_image_patches(x, [1, filter_size, filter_size, 1], strides, 4*[1], 'SAME', name="median_pool")
	patches = tf.expand_dims(patches, -1)
	medians = tf.contrib.distributions.percentile(patches, 50, axis=3, keep_dims=False)

	return medians

def compute_smooth_loss(input_of, output_q01):

	# Optical flow 2ch
	input_of = K.clip(input_of, min_value=-float('inf'), max_value=float('inf'))
	output_q01 = K.clip(output_q01, min_value=-float('inf'), max_value=float('inf'))

	# Epsilon prevents zero (loss goes inf)
	eps = K.epsilon()
	batch_size = tf.shape(input_of)[0]
	height = tf.shape(input_of)[1]
	width = tf.shape(input_of)[2]

	def gradient(pred):
		D_dy = pred[:, 1:, :, :] - pred[:, :-1, :, :]
		D_dx = pred[:, :, 1:, :] - pred[:, :, :-1, :]
		return D_dx, D_dy

	def disp_loss(pred_disp):
		dx, dy = gradient(pred_disp)
		dx2, dxdy = gradient(dx)
		dydx, dy2 = gradient(dy)
		loss = tf.reduce_mean(tf.abs(dx2)) + tf.reduce_mean(tf.abs(dxdy)) + tf.reduce_mean(tf.abs(dydx)) + tf.reduce_mean(tf.abs(dy2))
		return loss

	def custom_loss(y_true, y_pred):

		# output_t01 = y_pred
		output_t01 = K.clip(y_pred, min_value=-float('inf'), max_value=float('inf'))

		# Quaternion
		qw01, qx01, qy01, qz01 = output_q01[:,0], output_q01[:,1], output_q01[:,2], output_q01[:,3]
		q_norm = K.sqrt(K.square(qw01) + K.square(qx01) + K.square(qy01) + K.square(qz01) + eps)
		qw01, qx01, qy01, qz01 = qw01/q_norm, -qx01/q_norm, -qy01/q_norm, -qz01/q_norm

		# u,v tensor
		u_tensor = K.tile(K.arange(0, w, step=1, dtype='float32'), [h])
		u_tensor = K.reshape(u_tensor, [h, w])

		v_tensor = K.tile(K.arange(0, h, step=1, dtype='float32'), [w])
		v_tensor = K.transpose(K.reshape(v_tensor, [w, h]))

		# 3D location original
		X = K.sin(math.pi * u_tensor / h) * K.sin(math.pi * v_tensor / h)
		Y = K.cos(math.pi * u_tensor / h) * K.sin(math.pi * v_tensor / h)
		Z = K.cos(math.pi * v_tensor / h)
		XYZ = K.tile(K.expand_dims(K.reshape(K.stack([X, Y, Z]), [3, height * width]), 0), [batch_size, 1, 1])

		# Optical flow, FX/FY/FZ original
		FX = K.sin(math.pi * (u_tensor + input_of[:,:,:,0]) / h) * K.sin(math.pi * (v_tensor + input_of[:,:,:,1]) / h)
		FY = K.cos(math.pi * (u_tensor + input_of[:,:,:,0]) / h) * K.sin(math.pi * (v_tensor + input_of[:,:,:,1]) / h)
		FZ = K.cos(math.pi * (v_tensor + input_of[:,:,:,1]) / h)

		# 3D location derotated
		Rq = tf_batch_matrix([
			[qw01 * qw01 + qx01 * qx01 - qy01 * qy01 - qz01 * qz01, 2 * (qx01 * qy01 - qw01 * qz01), 2 * (qx01 * qz01 + qw01 * qy01)],
			[2 * (qx01 * qy01 + qw01 * qz01), qw01 * qw01 - qx01 * qx01 + qy01 * qy01 - qz01 * qz01, 2 * (qy01 * qz01 - qw01 * qx01)],
			[2 * (qx01 * qz01 - qw01 * qy01), 2 * (qy01 * qz01 + qw01 * qx01), qw01 * qw01 - qx01 * qx01 - qy01 * qy01 + qz01 * qz01]
		])

		XYZ_derotated = K.reshape(tf.matmul(Rq, XYZ), [batch_size, 3, height, width])
		X_derot_pred = XYZ_derotated[:, 0, :, :]
		Y_derot_pred = XYZ_derotated[:, 1, :, :]
		Z_derot_pred = XYZ_derotated[:, 2, :, :]

		Disp = K.sqrt((FX - X_derot_pred) ** 2 + (FY - Y_derot_pred) ** 2 + (FZ - Z_derot_pred) ** 2 + eps)

		tx01, ty01, tz01 = output_t01[:,0], output_t01[:,1], output_t01[:,2]
		t01 = K.sqrt(tx01**2 + ty01**2 + tz01**2 + eps)
		tx01, ty01, tz01 = -tx01/t01, -ty01/t01, -tz01/t01

		tx01 = K.reshape(K.tile(tx01, [h * w]), [batch_size, h, w])
		ty01 = K.reshape(K.tile(ty01, [h * w]), [batch_size, h, w])
		tz01 = K.reshape(K.tile(tz01, [h * w]), [batch_size, h, w])

		Omega = K.clip(X_derot_pred * tx01 + Y_derot_pred * ty01 + Z_derot_pred * tz01, min_value=-1.0+eps, max_value=1.0-eps)
		Omega = tf.acos(Omega)

		# Fix translation(t0->t1) scale to 1.0
		t_0 = 1.0
		Depth = t_0 * (K.abs(K.sin(Omega + Disp)) + eps) / (K.sin(Disp) + eps) + eps
		Depth = K.expand_dims(Depth, -1)
		smooth_loss = disp_loss(Depth)

		return smooth_loss

	return custom_loss

def epipolar_loss(input_of, output_q):

	# Optical flow 2ch
	input_of = K.clip(input_of, min_value=-float('inf'), max_value=float('inf'))
	output_q = K.clip(output_q, min_value=-float('inf'), max_value=float('inf'))

	# Epsilon prevents zero (loss goes inf)
	eps = K.epsilon()
	batch_size = tf.shape(input_of)[0]
	height = tf.shape(input_of)[1]
	width = tf.shape(input_of)[2]

	def custom_loss(y_true, y_pred):

		y_pred = K.clip(y_pred, min_value=-float('inf'), max_value=float('inf'))

		# Quaternion
		qw2, qx2, qy2, qz2 = output_q[:,0], output_q[:,1], output_q[:,2], output_q[:,3]
		q_norm = K.sqrt(K.square(qw2) + K.square(qx2) + K.square(qy2) + K.square(qz2) + eps)
		qw2, qx2, qy2, qz2 = qw2/q_norm, -qx2/q_norm, -qy2/q_norm, -qz2/q_norm

		# Translation
		epi_x, epi_y, epi_z = y_pred[:,0], y_pred[:,1], y_pred[:,2]
		epi_norm = K.sqrt(K.square(epi_x) + K.square(epi_y) + K.square(epi_z) + eps)
		epi_x, epi_y, epi_z = -epi_x/epi_norm, -epi_y/epi_norm, -epi_z/epi_norm

		# u,v tensor
		u_tensor = K.tile(K.arange(0, w, step=1, dtype='float32'), [h])
		u_tensor = K.reshape(u_tensor, [h, w])

		v_tensor = K.tile(K.arange(0, h, step=1, dtype='float32'), [w])
		v_tensor = K.transpose(K.reshape(v_tensor, [w, h]))

		# 3D location original
		X = K.sin(math.pi * u_tensor / h) * K.sin(math.pi * v_tensor / h)
		Y = K.cos(math.pi * u_tensor / h) * K.sin(math.pi * v_tensor / h)
		Z = K.cos(math.pi * v_tensor / h)
		XYZ = K.tile(K.expand_dims(K.reshape(K.stack([X, Y, Z]), [3, height * width]), 0), [batch_size, 1, 1])

		# Optical flow, FX/FY/FZ original
		FX = K.sin(math.pi * (u_tensor + input_of[:, :, :, 0]) / h) * K.sin(math.pi * (v_tensor + input_of[:, :, :, 1]) / h)
		FY = K.cos(math.pi * (u_tensor + input_of[:, :, :, 0]) / h) * K.sin(math.pi * (v_tensor + input_of[:, :, :, 1]) / h)
		FZ = K.cos(math.pi * (v_tensor + input_of[:, :, :, 1]) / h)

		# Optical flow, F original
		F = K.sqrt(K.square(FX) + K.square(FY) + K.square(FZ) + eps)

		# 3D location derotated
		Rq = tf_batch_matrix([
			[qw2 * qw2 + qx2 * qx2 - qy2 * qy2 - qz2 * qz2, 2 * (qx2 * qy2 - qw2 * qz2), 2 * (qx2 * qz2 + qw2 * qy2)],
			[2 * (qx2 * qy2 + qw2 * qz2), qw2 * qw2 - qx2 * qx2 + qy2 * qy2 - qz2 * qz2, 2 * (qy2 * qz2 - qw2 * qx2)],
			[2 * (qx2 * qz2 - qw2 * qy2), 2 * (qy2 * qz2 + qw2 * qx2), qw2 * qw2 - qx2 * qx2 - qy2 * qy2 + qz2 * qz2]
		])

		XYZ_derotated = K.reshape(tf.matmul(Rq, XYZ), [batch_size, 3, height, width])
		X_derot_pred = XYZ_derotated[:, 0, :, :]
		Y_derot_pred = XYZ_derotated[:, 1, :, :]
		Z_derot_pred = XYZ_derotated[:, 2, :, :]

		# Optical flow, FX/FY/FZ reprojected
		FX_pred = FX / (F + eps) - X_derot_pred
		FY_pred = FY / (F + eps) - Y_derot_pred
		FZ_pred = FZ / (F + eps) - Z_derot_pred

		# Weights
		W_kernel1 = K.tile(K.constant([1]), [int(w * h / 4) * batch_size])
		W_kernel1 = K.reshape(W_kernel1, [batch_size, int(w * h / 4)])
		W_kernel2 = K.tile(K.constant([1]), [int(w * h / 2) * batch_size])
		W_kernel2 = K.reshape(W_kernel2, [batch_size, int(w * h / 2)])

		W_kernel = K.concatenate([W_kernel1, W_kernel2], axis=-1)
		W_kernel = K.concatenate([W_kernel, W_kernel1], axis=-1)
		W_kernel = K.reshape(W_kernel, [batch_size, h, w])

		W = K.sqrt(1 - Z * Z + eps)
		W = W * W_kernel

		epi_x = K.expand_dims(K.expand_dims(epi_x, 1), 2)
		epi_y = K.expand_dims(K.expand_dims(epi_y, 1), 2)
		epi_z = K.expand_dims(K.expand_dims(epi_z, 1), 2)

		Nq_x, Nq_y, Nq_z = epi_y * Z_derot_pred - epi_z * Y_derot_pred, epi_z * X_derot_pred - epi_x * Z_derot_pred, \
						   epi_x * Y_derot_pred - epi_y * X_derot_pred
		Nf_x, Nf_y, Nf_z = FY_pred * Z_derot_pred - FZ_pred * Y_derot_pred, FZ_pred * X_derot_pred - FX_pred * Z_derot_pred, \
						   FX_pred * Y_derot_pred - FY_pred * X_derot_pred

		Nq_norm = K.sqrt(K.square(Nq_x) + K.square(Nq_y) + K.square(Nq_z) + eps)
		Nf_norm = K.sqrt(K.square(Nf_x) + K.square(Nf_y) + K.square(Nf_z) + eps)

		# 5DoF epipolar loss
		Epi_pred = tf.acos(K.clip((Nq_x * Nf_x + Nq_y * Nf_y + Nq_z * Nf_z) / (Nq_norm * Nf_norm + eps),
				min_value=-1.0 + eps, max_value=1.0 - eps)) * W

		# Loss function
		loss = Epi_pred

		return loss

	return custom_loss

def reconstruction_loss_SSIM(input_of, frame_t0, frame_t2, output_q01, output_t01, output_t02):

	# Optical flow 2ch
	input_of = K.clip(input_of, min_value=-float('inf'), max_value=float('inf'))
	frame_t0 = K.clip(frame_t0, min_value=-float('inf'), max_value=float('inf'))
	frame_t2 = K.clip(frame_t2, min_value=-float('inf'), max_value=float('inf'))
	output_q01 = K.clip(output_q01, min_value=-float('inf'), max_value=float('inf'))
	output_t02 = K.clip(output_t02, min_value=-float('inf'), max_value=float('inf'))
	output_t01 = K.clip(output_t01, min_value=-float('inf'), max_value=float('inf'))

	# Epsilon prevents zero (loss goes inf)
	eps = K.epsilon()
	batch_size = tf.shape(input_of)[0]
	height = tf.shape(input_of)[1]
	width = tf.shape(input_of)[2]

	def custom_loss(y_true, y_pred):

		# output_q02 = y_pred
		y_pred = K.clip(y_pred, min_value=-float('inf'), max_value=float('inf'))

		# Quaternion
		qw01, qx01, qy01, qz01 = output_q01[:,0], output_q01[:,1], output_q01[:,2], output_q01[:,3]
		q_norm = K.sqrt(K.square(qw01) + K.square(qx01) + K.square(qy01) + K.square(qz01) + eps)
		qw01, qx01, qy01, qz01 = qw01/q_norm, -qx01/q_norm, -qy01/q_norm, -qz01/q_norm

		# u,v tensor
		u_tensor = K.tile(K.arange(0, w, step=1, dtype='float32'), [h])
		u_tensor = K.reshape(u_tensor, [h, w])

		v_tensor = K.tile(K.arange(0, h, step=1, dtype='float32'), [w])
		v_tensor = K.transpose(K.reshape(v_tensor, [w, h]))

		# 3D location original
		X = K.sin(math.pi * u_tensor / h) * K.sin(math.pi * v_tensor / h)
		Y = K.cos(math.pi * u_tensor / h) * K.sin(math.pi * v_tensor / h)
		Z = K.cos(math.pi * v_tensor / h)
		XYZ = K.tile(K.expand_dims(K.reshape(K.stack([X, Y, Z]), [3, height * width]), 0), [batch_size, 1, 1])

		# Optical flow, FX/FY/FZ original
		FX = K.sin(math.pi * (u_tensor + input_of[:,:,:,0]) / h) * K.sin(math.pi * (v_tensor + input_of[:,:,:,1]) / h)
		FY = K.cos(math.pi * (u_tensor + input_of[:,:,:,0]) / h) * K.sin(math.pi * (v_tensor + input_of[:,:,:,1]) / h)
		FZ = K.cos(math.pi * (v_tensor + input_of[:,:,:,1]) / h)

		# 3D location derotated
		Rq = tf_batch_matrix([
			[qw01 * qw01 + qx01 * qx01 - qy01 * qy01 - qz01 * qz01, 2 * (qx01 * qy01 - qw01 * qz01), 2 * (qx01 * qz01 + qw01 * qy01)],
			[2 * (qx01 * qy01 + qw01 * qz01), qw01 * qw01 - qx01 * qx01 + qy01 * qy01 - qz01 * qz01, 2 * (qy01 * qz01 - qw01 * qx01)],
			[2 * (qx01 * qz01 - qw01 * qy01), 2 * (qy01 * qz01 + qw01 * qx01), qw01 * qw01 - qx01 * qx01 - qy01 * qy01 + qz01 * qz01]
		])

		XYZ_derotated = K.reshape(tf.matmul(Rq, XYZ), [batch_size, 3, height, width])
		X_derot_pred = XYZ_derotated[:, 0, :, :]
		Y_derot_pred = XYZ_derotated[:, 1, :, :]
		Z_derot_pred = XYZ_derotated[:, 2, :, :]

		Disp = K.sqrt((FX - X_derot_pred) ** 2 + (FY - Y_derot_pred) ** 2 + (FZ - Z_derot_pred) ** 2 + eps)

		tx01, ty01, tz01 = output_t01[:,0], output_t01[:,1], output_t01[:,2]
		t01 = K.sqrt(tx01**2 + ty01**2 + tz01**2 + eps)
		tx01, ty01, tz01 = -tx01/t01, -ty01/t01, -tz01/t01

		tx01 = K.reshape(K.tile(tx01, [h * w]), [batch_size, h, w])
		ty01 = K.reshape(K.tile(ty01, [h * w]), [batch_size, h, w])
		tz01 = K.reshape(K.tile(tz01, [h * w]), [batch_size, h, w])

		Omega = K.clip(X_derot_pred * tx01 + Y_derot_pred * ty01 + Z_derot_pred * tz01, min_value=-1.0+eps, max_value=1.0-eps)
		Omega = tf.acos(Omega)

		# Fix translation(t0->t1) scale to 1.0
		t_0 = 1.0
		Depth = t_0 * (K.abs(K.sin(Omega + Disp)) + eps) / (K.sin(Disp) + eps) + eps
		#Depth = tf.expand_dims(Depth, 3)

    # Gaussian filtered depth
		Depth = tf.expand_dims(Depth, 3)
		Depth = tf.squeeze(gaussian_blur(Depth, [7, 7], 2.0, 1), 3)

		# Median filtered depth
		#Depth = tf.expand_dims(Depth, 3)
		#Depth = tf.squeeze(median_blur(Depth, 5))

		# Reconstruction loss with depth
		qw02, qx02, qy02, qz02 = y_pred[:,0], y_pred[:,1], y_pred[:,2], y_pred[:,3]
		q02_norm = K.sqrt(K.square(qw02) + K.square(qx02) + K.square(qy02) + K.square(qz02) + eps)
		qw02, qx02, qy02, qz02 = qw02/(q02_norm + eps), qx02/(q02_norm + eps), qy02/(q02_norm + eps), qz02/(q02_norm + eps)

		img_syn = reconstruction.rotate_and_translate(frame_t0, Depth, qw02, qx02, qy02, qz02, output_t02[:,0], output_t02[:,1], output_t02[:,2])
		img_syn = K.reshape(img_syn, [batch_size, h, w, 3])

		#SSIM = tf.image.ssim_multiscale(frame_t2, img_syn, max_val=1.0, filter_size=6, power_factors=[0.4, 0.2, 0.2, 0.2, 0.0],filter_sigma=1.5, k1=0.01, k2=0.03)
		SSIM = tf.image.ssim(frame_t2, img_syn, max_val=1.0, filter_size=5, filter_sigma=1.5, k1=0.01, k2=0.03)
		loss = tf.reduce_mean(1.0 - SSIM)

		return loss

	return custom_loss

def reconstruction_loss_L1(input_of, frame_t0, frame_t2, output_q01, output_q02, output_t01):

	# Optical flow 2ch
	input_of = K.clip(input_of, min_value=-float('inf'), max_value=float('inf'))
	frame_t0 = K.clip(frame_t0, min_value=-float('inf'), max_value=float('inf'))
	frame_t2 = K.clip(frame_t2, min_value=-float('inf'), max_value=float('inf'))
	output_q01 = K.clip(output_q01, min_value=-float('inf'), max_value=float('inf'))
	output_q02 = K.clip(output_q02, min_value=-float('inf'), max_value=float('inf'))
	output_t01 = K.clip(output_t01, min_value=-float('inf'), max_value=float('inf'))

	# Epsilon prevents zero (loss goes inf)
	eps = K.epsilon()
	batch_size = tf.shape(input_of)[0]
	height = tf.shape(input_of)[1]
	width = tf.shape(input_of)[2]

	def custom_loss(y_true, y_pred):

		# output_t02 = y_pred
		y_pred = K.clip(y_pred, min_value=-float('inf'), max_value=float('inf'))

		# Quaternion
		qw01, qx01, qy01, qz01 = output_q01[:,0], output_q01[:,1], output_q01[:,2], output_q01[:,3]
		q_norm = K.sqrt(K.square(qw01) + K.square(qx01) + K.square(qy01) + K.square(qz01) + eps)
		qw01, qx01, qy01, qz01 = qw01/q_norm, -qx01/q_norm, -qy01/q_norm, -qz01/q_norm

		# u,v tensor
		u_tensor = K.tile(K.arange(0, w, step=1, dtype='float32'), [h])
		u_tensor = K.reshape(u_tensor, [h, w])

		v_tensor = K.tile(K.arange(0, h, step=1, dtype='float32'), [w])
		v_tensor = K.transpose(K.reshape(v_tensor, [w, h]))

		# 3D location original
		X = K.sin(math.pi * u_tensor / h) * K.sin(math.pi * v_tensor / h)
		Y = K.cos(math.pi * u_tensor / h) * K.sin(math.pi * v_tensor / h)
		Z = K.cos(math.pi * v_tensor / h)
		XYZ = K.tile(K.expand_dims(K.reshape(K.stack([X, Y, Z]), [3, height * width]), 0), [batch_size, 1, 1])

		# Optical flow, FX/FY/FZ original
		FX = K.sin(math.pi * (u_tensor + input_of[:,:,:,0]) / h) * K.sin(math.pi * (v_tensor + input_of[:,:,:,1]) / h)
		FY = K.cos(math.pi * (u_tensor + input_of[:,:,:,0]) / h) * K.sin(math.pi * (v_tensor + input_of[:,:,:,1]) / h)
		FZ = K.cos(math.pi * (v_tensor + input_of[:,:,:,1]) / h)

		# 3D location derotated
		Rq = tf_batch_matrix([
			[qw01 * qw01 + qx01 * qx01 - qy01 * qy01 - qz01 * qz01, 2 * (qx01 * qy01 - qw01 * qz01), 2 * (qx01 * qz01 + qw01 * qy01)],
			[2 * (qx01 * qy01 + qw01 * qz01), qw01 * qw01 - qx01 * qx01 + qy01 * qy01 - qz01 * qz01, 2 * (qy01 * qz01 - qw01 * qx01)],
			[2 * (qx01 * qz01 - qw01 * qy01), 2 * (qy01 * qz01 + qw01 * qx01), qw01 * qw01 - qx01 * qx01 - qy01 * qy01 + qz01 * qz01]
		])

		XYZ_derotated = K.reshape(tf.matmul(Rq, XYZ), [batch_size, 3, height, width])
		X_derot_pred = XYZ_derotated[:, 0, :, :]
		Y_derot_pred = XYZ_derotated[:, 1, :, :]
		Z_derot_pred = XYZ_derotated[:, 2, :, :]

		Disp = K.sqrt((FX - X_derot_pred) ** 2 + (FY - Y_derot_pred) ** 2 + (FZ - Z_derot_pred) ** 2 + eps)

		tx01, ty01, tz01 = output_t01[:,0], output_t01[:,1], output_t01[:,2]
		t01 = K.sqrt(tx01**2 + ty01**2 + tz01**2 + eps)
		tx01, ty01, tz01 = -tx01/t01, -ty01/t01, -tz01/t01

		tx01 = K.reshape(K.tile(tx01, [h * w]), [batch_size, h, w])
		ty01 = K.reshape(K.tile(ty01, [h * w]), [batch_size, h, w])
		tz01 = K.reshape(K.tile(tz01, [h * w]), [batch_size, h, w])

		Omega = K.clip(X_derot_pred * tx01 + Y_derot_pred * ty01 + Z_derot_pred * tz01, min_value=-1.0+eps, max_value=1.0-eps)
		Omega = tf.acos(Omega)

		# Fix translation(t0->t1) scale to 1.0
		t_0 = 1.0
		Depth = t_0 * (K.abs(K.sin(Omega + Disp)) + eps) / (K.sin(Disp) + eps) + eps
		#Depth = tf.expand_dims(Depth, 3)

		# Gaussian filtered depth
		Depth = tf.expand_dims(Depth, 3)
		Depth = tf.squeeze(gaussian_blur(Depth, [7, 7], 2.0, 1), 3)

		# Median filtered depth
		#Depth = tf.expand_dims(Depth, 3)
		#Depth = tf.squeeze(median_blur(Depth, 5))

		# Reconstruction loss with depth
		tx02, ty02, tz02 = y_pred[:,0], y_pred[:,1], y_pred[:,2]

		qw02, qx02, qy02, qz02 = output_q02[:,0], output_q02[:,1], output_q02[:,2], output_q02[:,3]
		q02_norm = K.sqrt(qw02**2 + qx02**2 + qy02**2 + qz02**2 + eps)
		qw02, qx02, qy02, qz02 = qw02/(q02_norm + eps), qx02/(q02_norm + eps), qy02/(q02_norm + eps), qz02/(q02_norm + eps)

		img_syn = reconstruction.rotate_and_translate(frame_t0, Depth, qw02, qx02, qy02, qz02, tx02, ty02, tz02)
		img_syn = K.reshape(img_syn, [batch_size, h, w, 3])

		# Weights
		W_kernel1 = K.tile(K.constant([1]), [int(w * h / 4) * batch_size])
		W_kernel1 = K.reshape(W_kernel1, [batch_size, int(w * h / 4)])
		W_kernel2 = K.tile(K.constant([1]), [int(w * h / 2) * batch_size])
		W_kernel2 = K.reshape(W_kernel2, [batch_size, int(w * h / 2)])

		W_kernel = K.concatenate([W_kernel1, W_kernel2], axis=-1)
		W_kernel = K.concatenate([W_kernel, W_kernel1], axis=-1)
		W_kernel = K.reshape(W_kernel, [batch_size, h, w])

		W = K.sqrt(1 - Z * Z)
		W = W * W_kernel
		W = K.expand_dims(W, 3)
		W = K.tile(W, [1, 1, 1, 3])

		loss = K.mean(K.abs(frame_t2 - img_syn) * W, axis=(1,2))

		return loss

	return custom_loss


def multi_scale_reconstruction_loss_L1(input_of, frame_t0, frame_t2, output_q01, output_q02, output_t01):

	# Optical flow 2ch
	input_of = K.clip(input_of, min_value=-float('inf'), max_value=float('inf'))
	frame_t0 = K.clip(frame_t0, min_value=-float('inf'), max_value=float('inf'))
	frame_t2 = K.clip(frame_t2, min_value=-float('inf'), max_value=float('inf'))
	output_q01 = K.clip(output_q01, min_value=-float('inf'), max_value=float('inf'))
	output_q02 = K.clip(output_q02, min_value=-float('inf'), max_value=float('inf'))
	output_t01 = K.clip(output_t01, min_value=-float('inf'), max_value=float('inf'))

	# Epsilon prevents zero (loss goes inf)
	eps = K.epsilon()
	batch_size = tf.shape(input_of)[0]
	height = tf.shape(input_of)[1]
	width = tf.shape(input_of)[2]
	img_height = 100
	img_width = 200

	def custom_loss(y_true, y_pred):

		# output_t02 = y_pred
		y_pred = K.clip(y_pred, min_value=-float('inf'), max_value=float('inf'))

		# Quaternion
		qw01, qx01, qy01, qz01 = output_q01[:,0], output_q01[:,1], output_q01[:,2], output_q01[:,3]
		q_norm = K.sqrt(K.square(qw01) + K.square(qx01) + K.square(qy01) + K.square(qz01) + eps)
		qw01, qx01, qy01, qz01 = qw01/q_norm, -qx01/q_norm, -qy01/q_norm, -qz01/q_norm

		# u,v tensor
		u_tensor = K.tile(K.arange(0, w, step=1, dtype='float32'), [h])
		u_tensor = K.reshape(u_tensor, [h, w])

		v_tensor = K.tile(K.arange(0, h, step=1, dtype='float32'), [w])
		v_tensor = K.transpose(K.reshape(v_tensor, [w, h]))

		# 3D location original
		X = K.sin(math.pi * u_tensor / h) * K.sin(math.pi * v_tensor / h)
		Y = K.cos(math.pi * u_tensor / h) * K.sin(math.pi * v_tensor / h)
		Z = K.cos(math.pi * v_tensor / h)
		XYZ = K.tile(K.expand_dims(K.reshape(K.stack([X, Y, Z]), [3, height * width]), 0), [batch_size, 1, 1])

		# Optical flow, FX/FY/FZ original
		FX = K.sin(math.pi * (u_tensor + input_of[:,:,:,0]) / h) * K.sin(math.pi * (v_tensor + input_of[:,:,:,1]) / h)
		FY = K.cos(math.pi * (u_tensor + input_of[:,:,:,0]) / h) * K.sin(math.pi * (v_tensor + input_of[:,:,:,1]) / h)
		FZ = K.cos(math.pi * (v_tensor + input_of[:,:,:,1]) / h)

		# 3D location derotated
		Rq = tf_batch_matrix([
			[qw01 * qw01 + qx01 * qx01 - qy01 * qy01 - qz01 * qz01, 2 * (qx01 * qy01 - qw01 * qz01), 2 * (qx01 * qz01 + qw01 * qy01)],
			[2 * (qx01 * qy01 + qw01 * qz01), qw01 * qw01 - qx01 * qx01 + qy01 * qy01 - qz01 * qz01, 2 * (qy01 * qz01 - qw01 * qx01)],
			[2 * (qx01 * qz01 - qw01 * qy01), 2 * (qy01 * qz01 + qw01 * qx01), qw01 * qw01 - qx01 * qx01 - qy01 * qy01 + qz01 * qz01]
		])

		XYZ_derotated = K.reshape(tf.matmul(Rq, XYZ), [batch_size, 3, height, width])
		X_derot_pred = XYZ_derotated[:, 0, :, :]
		Y_derot_pred = XYZ_derotated[:, 1, :, :]
		Z_derot_pred = XYZ_derotated[:, 2, :, :]

		Disp = K.sqrt((FX - X_derot_pred) ** 2 + (FY - Y_derot_pred) ** 2 + (FZ - Z_derot_pred) ** 2 + eps)

		tx01, ty01, tz01 = output_t01[:,0], output_t01[:,1], output_t01[:,2]
		t01 = K.sqrt(tx01**2 + ty01**2 + tz01**2 + eps)
		tx01, ty01, tz01 = -tx01/t01, -ty01/t01, -tz01/t01

		tx01 = K.reshape(K.tile(tx01, [h * w]), [batch_size, h, w])
		ty01 = K.reshape(K.tile(ty01, [h * w]), [batch_size, h, w])
		tz01 = K.reshape(K.tile(tz01, [h * w]), [batch_size, h, w])

		Omega = K.clip(X_derot_pred * tx01 + Y_derot_pred * ty01 + Z_derot_pred * tz01, min_value=-1.0+eps, max_value=1.0-eps)
		Omega = tf.acos(Omega)

		# Fix translation(t0->t1) scale to 1.0
		t_0 = 1.0
		Depth = t_0 * (K.abs(K.sin(Omega + Disp)) + eps) / (K.sin(Disp) + eps) + eps

		# Gaussian filtered depth
		Depth = tf.expand_dims(Depth, 3)
		Depth = gaussian_blur(Depth, [7, 7], 2.0, 1)

		# Reconstruction loss with depth
		tx02, ty02, tz02 = y_pred[:,0], y_pred[:,1], y_pred[:,2]

		qw02, qx02, qy02, qz02 = output_q02[:,0], output_q02[:,1], output_q02[:,2], output_q02[:,3]
		q02_norm = K.sqrt(qw02**2 + qx02**2 + qy02**2 + qz02**2 + eps)
		qw02, qx02, qy02, qz02 = qw02/(q02_norm + eps), qx02/(q02_norm + eps), qy02/(q02_norm + eps), qz02/(q02_norm + eps)

		# Resize for multi-scale
		frame_t0_1 = tf.image.resize_area(frame_t0, [int(img_height / (2**3)), int(img_width / (2**3))])  # 25x13
		frame_t0_2 = tf.image.resize_area(frame_t0, [int(img_height / (2**2)), int(img_width / (2**2))])  # 50x25
		frame_t0_3 = tf.image.resize_area(frame_t0, [int(img_height / (2**1)), int(img_width / (2**1))])  # 100x50
		frame_t0_4 = tf.image.resize_area(frame_t0, [int(img_height / (2**0)), int(img_width / (2**0))])  # 200x100

		frame_t2_1 = tf.image.resize_area(frame_t2, [int(img_height / (2**3)), int(img_width / (2**3))])
		frame_t2_2 = tf.image.resize_area(frame_t2, [int(img_height / (2**2)), int(img_width / (2**2))])
		frame_t2_3 = tf.image.resize_area(frame_t2, [int(img_height / (2**1)), int(img_width / (2**1))])
		frame_t2_4 = tf.image.resize_area(frame_t2, [int(img_height / (2**0)), int(img_width / (2**0))])

		depth_1 = tf.image.resize_nearest_neighbor(Depth, [int(img_height / (2**3)), int(img_width / (2**3))])
		depth_2 = tf.image.resize_nearest_neighbor(Depth, [int(img_height / (2**2)), int(img_width / (2**2))])
		depth_3 = tf.image.resize_nearest_neighbor(Depth, [int(img_height / (2**1)), int(img_width / (2**1))])
		depth_4 = tf.image.resize_nearest_neighbor(Depth, [int(img_height / (2**0)), int(img_width / (2**0))])

		# Synthetic images for multi-scale
		img_syn_1 = reconstruction.rotate_and_translate(frame_t0_1, tf.squeeze(depth_1, 3), qw02, qx02, qy02, qz02, tx02, ty02, tz02)
		img_syn_1 = tf.reshape(img_syn_1, [batch_size, int(img_height / (2**3)), int(img_width / (2**3)), 3])
		img_syn_2 = reconstruction.rotate_and_translate(frame_t0_2, tf.squeeze(depth_2, 3), qw02, qx02, qy02, qz02, tx02, ty02, tz02)
		img_syn_2 = tf.reshape(img_syn_2, [batch_size, int(img_height / (2**2)), int(img_width / (2**2)), 3])
		img_syn_3 = reconstruction.rotate_and_translate(frame_t0_3, tf.squeeze(depth_3, 3), qw02, qx02, qy02, qz02, tx02, ty02, tz02)
		img_syn_3 = tf.reshape(img_syn_3, [batch_size, int(img_height / (2**1)), int(img_width / (2**1)), 3])
		img_syn_4 = reconstruction.rotate_and_translate(frame_t0_4, tf.squeeze(depth_4, 3), qw02, qx02, qy02, qz02, tx02, ty02, tz02)
		img_syn_4 = tf.reshape(img_syn_4, [batch_size, int(img_height / (2**0)), int(img_width / (2**0)), 3])

		# Multi-distortion weights
		v_1 = K.tile(K.arange(0, int(img_height / (2**3)), step=1, dtype='float32'), [int(img_width / (2**3))])
		v_1 = K.transpose(K.reshape(v_1, [int(img_width / (2**3)), int(img_height / (2**3))]))
		v_2 = K.tile(K.arange(0, int(img_height / (2**2)), step=1, dtype='float32'), [int(img_width / (2**2))])
		v_2 = K.transpose(K.reshape(v_2, [int(img_width / (2**2)), int(img_height / (2**2))]))
		v_3 = K.tile(K.arange(0, int(img_height / (2**1)), step=1, dtype='float32'), [int(img_width / (2**1))])
		v_3 = K.transpose(K.reshape(v_3, [int(img_width / (2**1)), int(img_height / (2**1))]))
		v_4 = K.tile(K.arange(0, int(img_height / (2**0)), step=1, dtype='float32'), [int(img_width / (2**0))])
		v_4 = K.transpose(K.reshape(v_4, [int(img_width / (2**0)), int(img_height / (2**0))]))

		z_1 = K.cos(math.pi * v_1 / int(img_height / (2**3)))
		z_2 = K.cos(math.pi * v_2 / int(img_height / (2**2)))
		z_3 = K.cos(math.pi * v_3 / int(img_height / (2**1)))
		z_4 = K.cos(math.pi * v_4 / int(img_height / (2**0)))

		w_1 = K.sqrt(1-z_1**2)
		w_1 = K.tile(K.expand_dims(w_1, -1), [1, 1, 3])
		w_2 = K.sqrt(1-z_2**2)
		w_2 = K.tile(K.expand_dims(w_2, -1), [1, 1, 3])
		w_3 = K.sqrt(1-z_3**2)
		w_3 = K.tile(K.expand_dims(w_3, -1), [1, 1, 3])
		w_4 = K.sqrt(1-z_4**2)
		w_4 = K.tile(K.expand_dims(w_4, -1), [1, 1, 3])

		# Multi-scale photometric reprojection loss
		loss_1 = tf.reduce_mean(tf.abs(frame_t2_1 - img_syn_1) * w_1, axis=(1,2))
		loss_2 = tf.reduce_mean(tf.abs(frame_t2_2 - img_syn_2) * w_2, axis=(1,2))
		loss_3 = tf.reduce_mean(tf.abs(frame_t2_3 - img_syn_3) * w_3, axis=(1,2))
		loss_4 = tf.reduce_mean(tf.abs(frame_t2_4 - img_syn_4) * w_4, axis=(1,2))

		loss = loss_1 + loss_2 + loss_3 + loss_4

		return loss

	return custom_loss


def moment_loss(input_of):  # y_pred = quaternion output
	# Optical flow 2ch
	input_of = K.clip(input_of, min_value=-float('inf'), max_value=float('inf'))

	# Epsilon prevents zero (loss goes inf)
	eps = K.epsilon()
	batch_size = tf.shape(input_of)[0]
	height = tf.shape(input_of)[1]
	width = tf.shape(input_of)[2]

	def custom_loss(y_true, y_pred):

		y_pred = K.clip(y_pred, min_value=-float('inf'), max_value=float('inf'))

		# Quaternion
		qw2, qx2, qy2, qz2 = y_pred[:,0], y_pred[:,1], y_pred[:,2], y_pred[:,3]
		q_norm = K.sqrt(K.square(qw2) + K.square(qx2) + K.square(qy2) + K.square(qz2) + eps)
		qw2, qx2, qy2, qz2 = qw2/q_norm, -qx2/q_norm, -qy2/q_norm, -qz2/q_norm

		# u,v tensor
		u_tensor = K.tile(K.arange(0, w, step=1, dtype='float32'), [h])
		u_tensor = K.reshape(u_tensor, [h, w])

		v_tensor = K.tile(K.arange(0, h, step=1, dtype='float32'), [w])
		v_tensor = K.transpose(K.reshape(v_tensor, [w, h]))

		# 3D location original
		X = K.sin(math.pi * u_tensor / h) * K.sin(math.pi * v_tensor / h)
		Y = K.cos(math.pi * u_tensor / h) * K.sin(math.pi * v_tensor / h)
		Z = K.cos(math.pi * v_tensor / h)
		XYZ = K.tile(K.expand_dims(K.reshape(K.stack([X, Y, Z]), [3, height * width]), 0), [batch_size, 1, 1])

		# Optical flow, FX/FY/FZ original
		FX = K.sin(math.pi * (u_tensor + input_of[:, :, :, 0]) / h) * K.sin(math.pi * (v_tensor + input_of[:, :, :, 1]) / h)
		FY = K.cos(math.pi * (u_tensor + input_of[:, :, :, 0]) / h) * K.sin(math.pi * (v_tensor + input_of[:, :, :, 1]) / h)
		FZ = K.cos(math.pi * (v_tensor + input_of[:, :, :, 1]) / h)

		# Optical flow, F original
		F = K.sqrt(K.square(FX) + K.square(FY) + K.square(FZ) + eps)

		# 3D location derotated
		Rq = tf_batch_matrix([
			[qw2 * qw2 + qx2 * qx2 - qy2 * qy2 - qz2 * qz2, 2 * (qx2 * qy2 - qw2 * qz2), 2 * (qx2 * qz2 + qw2 * qy2)],
			[2 * (qx2 * qy2 + qw2 * qz2), qw2 * qw2 - qx2 * qx2 + qy2 * qy2 - qz2 * qz2, 2 * (qy2 * qz2 - qw2 * qx2)],
			[2 * (qx2 * qz2 - qw2 * qy2), 2 * (qy2 * qz2 + qw2 * qx2), qw2 * qw2 - qx2 * qx2 - qy2 * qy2 + qz2 * qz2]
		])

		XYZ_derotated = K.reshape(tf.matmul(Rq, XYZ), [batch_size, 3, height, width])
		X_derot_pred = XYZ_derotated[:, 0, :, :]
		Y_derot_pred = XYZ_derotated[:, 1, :, :]
		Z_derot_pred = XYZ_derotated[:, 2, :, :]

		# Optical flow, FX/FY/FZ reprojected
		FX_pred = FX / (F + eps) - X_derot_pred
		FY_pred = FY / (F + eps) - Y_derot_pred
		FZ_pred = FZ / (F + eps) - Z_derot_pred

		# Weights
		W_kernel1 = K.tile(K.constant([0]), [int(w * h / 4) * batch_size])
		W_kernel1 = K.reshape(W_kernel1, [batch_size, int(w * h / 4)])
		W_kernel2 = K.tile(K.constant([1]), [int(w * h / 2) * batch_size])
		W_kernel2 = K.reshape(W_kernel2, [batch_size, int(w * h / 2)])

		W_kernel = K.concatenate([W_kernel1, W_kernel2], axis=-1)
		W_kernel = K.concatenate([W_kernel, W_kernel1], axis=-1)
		W_kernel = K.reshape(W_kernel, [batch_size, h, w])

		W = K.sqrt(1 - Z * Z + eps)
		W = W * W_kernel

		# Directional moment with weights, MX/MY/MZ
		Moment_X_pred = (Y_derot_pred * FZ_pred - Z_derot_pred * FY_pred) * W
		Moment_Y_pred = (Z_derot_pred * FX_pred - X_derot_pred * FZ_pred) * W
		Moment_Z_pred = (X_derot_pred * FY_pred - Y_derot_pred * FX_pred) * W

		Moment_X_pred_sum = K.sum(Moment_X_pred, axis=(1,2))
		Moment_Y_pred_sum = K.sum(Moment_Y_pred, axis=(1,2))
		Moment_Z_pred_sum = K.sum(Moment_Z_pred, axis=(1,2))

		Moment_XYZ_pred_sum = K.sqrt(K.square(Moment_X_pred_sum) + K.square(Moment_Y_pred_sum) + K.square(Moment_Z_pred_sum) + eps) / (w*h/2)

		# Loss function
		loss = Moment_XYZ_pred_sum

		return loss

	return custom_loss

def supervision(y_true, y_pred):
	loss = K.sqrt(K.sum(K.square(y_true[:, :] - y_pred[:, :]), axis=1, keepdims=True))

	return loss

def loss_dummy(y_true, y_pred):
	y_pred = K.clip(y_pred, min_value=-float('inf'), max_value=float('inf'))

	return y_pred

def read_npy(input_path, height, width, dim):
	input = np.load(input_path)
	input = tf.convert_to_tensor(input, dtype=tf.float32)
	input = tf.reshape(input, [height, width, dim])

	return tf.expand_dims(input, 0)

def read_image(image_path, shape):
	if image_path.lower().endswith("png"):
		image = tf.image.decode_png(tf.io.read_file(image_path))
	else:
		image = tf.image.decode_jpeg(tf.io.read_file(image_path))
	image = tf.image.convert_image_dtype(image, tf.float32)
	image = tf.image.resize(image, shape, tf.image.ResizeMethod.AREA)

	return tf.expand_dims(image, 0)

def epipolar_loss_check(inputs, qw, qx, qy, qz, tx, ty, tz):

	# Optical flow 2ch
	input_of = K.clip(inputs, min_value=-float('inf'), max_value=float('inf'))

	# Epsilon prevents zero (loss goes inf)
	eps = K.epsilon()
	batch_size = tf.shape(inputs)[0]
	height = tf.shape(inputs)[1]
	width = tf.shape(inputs)[2]

	# Quaternion
	qw2, qx2, qy2, qz2 = qw, qx, qy, qz
	q_norm = K.sqrt(K.square(qw2) + K.square(qx2) + K.square(qy2) + K.square(qz2) + eps)
	qw2, qx2, qy2, qz2 = qw2/q_norm, -qx2/q_norm, -qy2/q_norm, -qz2/q_norm

	# Translation
	epi_x, epi_y, epi_z = tx, ty, tz
	epi_norm = K.sqrt(K.square(epi_x) + K.square(epi_y) + K.square(epi_z) + eps)
	epi_x, epi_y, epi_z = -epi_x/epi_norm, -epi_y/epi_norm, -epi_z/epi_norm

	# u,v tensor
	u_tensor = K.tile(K.arange(0, w, step=1, dtype='float32'), [h])
	u_tensor = K.reshape(u_tensor, [h,w])

	v_tensor = K.tile(K.arange(0, h, step=1, dtype='float32'), [w])
	v_tensor = K.transpose(K.reshape(v_tensor, [w,h]))

	# 3D location original
	X = K.sin(math.pi * u_tensor / h) * K.sin(math.pi * v_tensor / h)
	Y = K.cos(math.pi * u_tensor / h) * K.sin(math.pi * v_tensor / h)
	Z = K.cos(math.pi * v_tensor / h)
	XYZ = K.tile(K.expand_dims(K.reshape(K.stack([X, Y, Z]), [3, height * width]), 0), [batch_size, 1, 1])

	# Optical flow, FX/FY/FZ original
	FX = K.sin(math.pi * (u_tensor + input_of[:,:,:,0]) / h) * K.sin(math.pi * (v_tensor + input_of[:,:,:,1]) / h)
	FY = K.cos(math.pi * (u_tensor + input_of[:,:,:,0]) / h) * K.sin(math.pi * (v_tensor + input_of[:,:,:,1]) / h)
	FZ = K.cos(math.pi * (v_tensor + input_of[:,:,:,1]) / h)

	# Optical flow, F original
	F = K.sqrt(K.square(FX) + K.square(FY) + K.square(FZ) + eps)

	# 3D location derotated
	Rq = tf_batch_matrix([
		[qw2*qw2 + qx2*qx2 - qy2*qy2 - qz2*qz2, 2*(qx2*qy2 - qw2*qz2), 2*(qx2*qz2 + qw2*qy2)],
		[2*(qx2*qy2 + qw2*qz2), qw2*qw2 - qx2*qx2 + qy2*qy2 - qz2*qz2, 2*(qy2*qz2 - qw2*qx2)],
		[2*(qx2*qz2 - qw2*qy2), 2*(qy2*qz2 + qw2*qx2), qw2*qw2 - qx2*qx2 - qy2*qy2 + qz2*qz2]
	])

	XYZ_derotated = K.reshape(tf.matmul(Rq, XYZ), [batch_size, 3, height, width])
	X_derot_pred = XYZ_derotated[:, 0, :, :]
	Y_derot_pred = XYZ_derotated[:, 1, :, :]
	Z_derot_pred = XYZ_derotated[:, 2, :, :]

	# Optical flow, FX/FY/FZ reprojected
	FX_pred = FX/(F+eps) - X_derot_pred
	FY_pred = FY/(F+eps) - Y_derot_pred
	FZ_pred = FZ/(F+eps) - Z_derot_pred

	# Weights
	W_kernel1 = K.tile(K.constant([0]), [int(w*h/4) * 3])
	W_kernel1 = K.reshape(W_kernel1, [3, int(w*h/4)])
	W_kernel2 = K.tile(K.constant([1]), [int(w*h/2) * 3])
	W_kernel2 = K.reshape(W_kernel2, [3, int(w*h/2)])

	W_kernel = K.concatenate([W_kernel1, W_kernel2], axis=-1)
	W_kernel = K.concatenate([W_kernel, W_kernel1], axis=-1)
	W_kernel = K.reshape(W_kernel, [3,h,w])

	W = K.sqrt(1-Z*Z + eps)
	W = W * W_kernel

	epi_x = K.expand_dims(K.expand_dims(epi_x, 1), 2)
	epi_y = K.expand_dims(K.expand_dims(epi_y, 1), 2)
	epi_z = K.expand_dims(K.expand_dims(epi_z, 1), 2)

	Nq_x, Nq_y, Nq_z = epi_y*Z_derot_pred - epi_z*Y_derot_pred, epi_z*X_derot_pred - epi_x*Z_derot_pred, \
		epi_x*Y_derot_pred - epi_y*X_derot_pred
	Nf_x, Nf_y, Nf_z = FY_pred*Z_derot_pred - FZ_pred*Y_derot_pred, FZ_pred*X_derot_pred - FX_pred*Z_derot_pred, \
		FX_pred*Y_derot_pred - FY_pred*X_derot_pred

	Nq_norm = K.sqrt(K.square(Nq_x) + K.square(Nq_y) + K.square(Nq_z) + eps)
	Nf_norm = K.sqrt(K.square(Nf_x) + K.square(Nf_y) + K.square(Nf_z) + eps)

	# 5DoF epipolar loss
	Epi_pred = tf.acos(K.clip((Nq_x*Nf_x + Nq_y*Nf_y + Nq_z*Nf_z) / (Nq_norm*Nf_norm + eps), min_value=-1.0+eps, max_value=1.0-eps)) * W
	Epi_pred = K.square(Epi_pred)
	Epi_pred = K.sum(Epi_pred, axis=(1,2))

	# Loss function
	loss = Epi_pred

	return loss
