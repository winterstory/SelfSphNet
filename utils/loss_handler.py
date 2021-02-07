import math
import tensorflow as tf
import utils.helper as helper
import utils.reconstruction as reconstructor

from keras import backend as K


def get_epipolar_loss(optical_flow, output_quaternion):
    # Optical flow 2ch
    optical_flow = K.clip(
        optical_flow,
        min_value = -float('inf'),
        max_value = float('inf'))
    output_quaternion = K.clip(
        output_quaternion,
        min_value = -float('inf'),
        max_value = float('inf'))

    # Epsilon prevents zero (loss goes inf)
    eps = K.epsilon()
    w, h = 200, 100
    tf_batch_size = tf.shape(optical_flow)[0]
    tf_height = tf.shape(optical_flow)[1]
    tf_width = tf.shape(optical_flow)[2]


    def custom_loss(y_true, y_pred):
        y_pred = K.clip(y_pred, min_value=-float('inf'), max_value=float('inf'))

        # Quaternion
        qw = output_quaternion[:,0]
        qx = output_quaternion[:,1]
        qy = output_quaternion[:,2]
        qz = output_quaternion[:,3]
        q_norm = K.sqrt(K.square(qw) + K.square(qx) + K.square(qy) + K.square(qz) + eps)
        qw, qx, qy, qz = qw / q_norm, -qx / q_norm, -qy / q_norm, -qz / q_norm

        # Translation
        epi_x, epi_y, epi_z = y_pred[:,0], y_pred[:,1], y_pred[:,2]
        epi_norm = K.sqrt(K.square(epi_x) + K.square(epi_y) + K.square(epi_z) + eps)
        epi_x, epi_y, epi_z = -epi_x / epi_norm, -epi_y / epi_norm, -epi_z / epi_norm

        # Optical flow, F original
        FX = helper.get_3d_optical_flow(w, h, optical_flow)[0]
        FY = helper.get_3d_optical_flow(w, h, optical_flow)[1]
        FZ = helper.get_3d_optical_flow(w, h, optical_flow)[2]
        F = K.sqrt(K.square(FX) + K.square(FY) + K.square(FZ) + eps)

        # 3D location derotated
        qw2 = qw * qw
        qx2 = qx * qx
        qy2 = qy * qy
        qz2 = qz * qz
        Rq = helper.get_tf_batch_matrix([
            [qw2 + qx2 - qy2 - qz2, 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
            [2 * (qx * qy + qw * qz), qw2 - qx2 + qy2 - qz2, 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), qw2 - qx2 - qy2 + qz2]])

        XYZ = helper.get_3d_optical_flow(w, h, optical_flow)[3]
        XYZ_derotated = K.reshape(tf.matmul(Rq, XYZ), [tf_batch_size, 3, tf_height, tf_width])
        X_derot_pred = XYZ_derotated[:, 0, :, :]
        Y_derot_pred = XYZ_derotated[:, 1, :, :]
        Z_derot_pred = XYZ_derotated[:, 2, :, :]

        # Optical flow, FX/FY/FZ reprojected
        FX_pred = FX / (F + eps) - X_derot_pred
        FY_pred = FY / (F + eps) - Y_derot_pred
        FZ_pred = FZ / (F + eps) - Z_derot_pred

        # Weights
        W_kernel1 = K.tile(K.constant([1]), [int(w * h / 4) * tf_batch_size])
        W_kernel1 = K.reshape(W_kernel1, [tf_batch_size, int(w * h / 4)])
        W_kernel2 = K.tile(K.constant([1]), [int(w * h / 2) * tf_batch_size])
        W_kernel2 = K.reshape(W_kernel2, [tf_batch_size, int(w * h / 2)])

        W_kernel = K.concatenate([W_kernel1, W_kernel2], axis=-1)
        W_kernel = K.concatenate([W_kernel, W_kernel1], axis=-1)
        W_kernel = K.reshape(W_kernel, [tf_batch_size, h, w])

        Z = helper.get_3d_optical_flow(w, h, optical_flow)[4]
        W = K.sqrt(1 - Z * Z + eps)
        W = W * W_kernel

        epi_x = K.expand_dims(K.expand_dims(epi_x, 1), 2)
        epi_y = K.expand_dims(K.expand_dims(epi_y, 1), 2)
        epi_z = K.expand_dims(K.expand_dims(epi_z, 1), 2)

        Nq_x = epi_y * Z_derot_pred - epi_z * Y_derot_pred
        Nq_y = epi_z * X_derot_pred - epi_x * Z_derot_pred
        Nq_z = epi_x * Y_derot_pred - epi_y * X_derot_pred

        Nf_x = FY_pred * Z_derot_pred - FZ_pred * Y_derot_pred
        Nf_y = FZ_pred * X_derot_pred - FX_pred * Z_derot_pred
        Nf_z = FX_pred * Y_derot_pred - FY_pred * X_derot_pred

        Nq_norm = K.sqrt(K.square(Nq_x) + K.square(Nq_y) + K.square(Nq_z) + eps)
        Nf_norm = K.sqrt(K.square(Nf_x) + K.square(Nf_y) + K.square(Nf_z) + eps)

        # 5DoF epipolar loss
        Epi_pred = tf.acos(K.clip(
            (Nq_x * Nf_x + Nq_y * Nf_y + Nq_z * Nf_z) / (Nq_norm * Nf_norm + eps),
            min_value = -1.0 + eps,
            max_value = 1.0 - eps)) * W

        # Loss function
        out_loss = Epi_pred
        return out_loss

    return custom_loss


def get_ssim_loss(optical_flow, frame_t0, frame_t2, output_q01, output_t01, output_t02):
    # Optical flow 2ch
    optical_flow = K.clip(optical_flow, min_value=-float('inf'), max_value=float('inf'))
    frame_t0 = K.clip(frame_t0, min_value=-float('inf'), max_value=float('inf'))
    frame_t2 = K.clip(frame_t2, min_value=-float('inf'), max_value=float('inf'))
    output_q01 = K.clip(output_q01, min_value=-float('inf'), max_value=float('inf'))
    output_t02 = K.clip(output_t02, min_value=-float('inf'), max_value=float('inf'))
    output_t01 = K.clip(output_t01, min_value=-float('inf'), max_value=float('inf'))

    # Epsilon prevents zero (loss goes inf)
    eps = K.epsilon()
    w, h = 200, 100
    tf_batch_size = tf.shape(optical_flow)[0]
    tf_height = tf.shape(optical_flow)[1]
    tf_width = tf.shape(optical_flow)[2]


    def custom_loss(y_true, y_pred):
        # output_q02 = y_pred
        y_pred = K.clip(y_pred, min_value=-float('inf'), max_value=float('inf'))

        # Quaternion
        qw01 = output_q01[:,0]
        qx01 = output_q01[:,1]
        qy01 = output_q01[:,2]
        qz01 = output_q01[:,0]
        q_norm = K.sqrt(K.square(qw01) + K.square(qx01) + K.square(qy01) + K.square(qz01) + eps)
        qw01, qx01, qy01, qz01 = qw01 / q_norm, -qx01 / q_norm, -qy01 / q_norm, -qz01 / q_norm

        FX = helper.get_3d_optical_flow(w, h, optical_flow)[0]
        FY = helper.get_3d_optical_flow(w, h, optical_flow)[1]
        FZ = helper.get_3d_optical_flow(w, h, optical_flow)[2]
        XYZ = helper.get_3d_optical_flow(w, h, optical_flow)[3]

        # 3D location derotated
        R11 = qw01 ** 2 + qx01 ** 2 - qy01 ** 2 - qz01 ** 2
        R12 = 2 * (qx01 * qy01 - qw01 * qz01)
        R13 = 2 * (qx01 * qz01 + qw01 * qy01)
        R21 = 2 * (qx01 * qy01 + qw01 * qz01)
        R22 = qw01 ** 2 - qx01 ** 2 + qy01 ** 2 - qz01 ** 2
        R23 = 2 * (qy01 * qz01 - qw01 * qx01)
        R31 = 2 * (qx01 * qz01 - qw01 * qy01)
        R32 = 2 * (qy01 * qz01 + qw01 * qx01)
        R33 = qw01 ** 2 - qx01 ** 2 - qy01 ** 2 + qz01 ** 2
        Rq = helper.get_tf_batch_matrix([
            [R11, R12, R13],
            [R21, R22, R23],
            [R31, R32, R33]])

        XYZ_derotated = K.reshape(
            tf.matmul(Rq, XYZ),
            [tf_batch_size, 3, tf_height, tf_width])
        X_derot_pred = XYZ_derotated[:, 0, :, :]
        Y_derot_pred = XYZ_derotated[:, 1, :, :]
        Z_derot_pred = XYZ_derotated[:, 2, :, :]

        disparity = K.sqrt((FX - X_derot_pred) ** 2 + (FY - Y_derot_pred) ** 2 + (FZ - Z_derot_pred) ** 2 + eps)

        tx01, ty01, tz01 = output_t01[:,0], output_t01[:,1], output_t01[:,2]
        t01 = K.sqrt(tx01**2 + ty01**2 + tz01**2 + eps)
        tx01, ty01, tz01 = -tx01/t01, -ty01/t01, -tz01/t01

        tx01 = K.reshape(K.tile(tx01, [h * w]), [tf_batch_size, h, w])
        ty01 = K.reshape(K.tile(ty01, [h * w]), [tf_batch_size, h, w])
        tz01 = K.reshape(K.tile(tz01, [h * w]), [tf_batch_size, h, w])

        omega = K.clip(
            X_derot_pred * tx01 + Y_derot_pred * ty01 + Z_derot_pred * tz01,
            min_value = -1.0 + eps,
            max_value = 1.0 - eps)
        omega = tf.acos(omega)

        # Fix translation(t0->t1) scale to 1.0
        fixed_translation_scale = 1.0
        depth = fixed_translation_scale * (K.abs(K.sin(omega + disparity)) + eps) /\
                (K.sin(disparity) + eps) + eps
        #depth = tf.expand_dims(depth, 3)

        # Gaussian filtered depth
        depth = tf.expand_dims(depth, 3)
        depth = tf.squeeze(helper.apply_gaussian_blur(depth, [7, 7], 2.0, 1), 3)
        # Median filtered depth
        #depth = tf.expand_dims(depth, 3)
        #depth = tf.squeeze(apply_median_blur(depth, 5))

        # Reconstruction loss with depth
        qw02, qx02, qy02, qz02 = y_pred[:,0], y_pred[:,1], y_pred[:,2], y_pred[:,3]
        q02_norm = K.sqrt(K.square(qw02) + K.square(qx02) + K.square(qy02) + K.square(qz02) + eps)
        qw02 = qw02 / (q02_norm + eps)
        qx02 = qx02 / (q02_norm + eps)
        qy02 = qy02 / (q02_norm + eps)
        qz02 = qz02 / (q02_norm + eps)

        synthesized_image = reconstructor.rotate_and_translate(
            frame_t0,
            depth,
            qw02,
            qx02,
            qy02,
            qz02,
            output_t02[:,0],
            output_t02[:,1],
            output_t02[:,2])
        synthesized_image = K.reshape(synthesized_image, [tf_batch_size, h, w, 3])

        SSIM = tf.image.ssim(
            frame_t2,
            synthesized_image,
            max_val = 1.0,
            filter_size = 5,
            filter_sigma = 1.5,
            k1 = 0.01,
            k2 = 0.03)
        SSIM_loss = tf.reduce_mean(1.0 - SSIM)
        return SSIM_loss

    return custom_loss


def get_l1_loss(optical_flow, frame_t0, frame_t2, output_q01, output_q02, output_t01):
    # Optical flow 2ch
    optical_flow = K.clip(optical_flow, min_value=-float('inf'), max_value=float('inf'))
    frame_t0 = K.clip(frame_t0, min_value=-float('inf'), max_value=float('inf'))
    frame_t2 = K.clip(frame_t2, min_value=-float('inf'), max_value=float('inf'))
    output_q01 = K.clip(output_q01, min_value=-float('inf'), max_value=float('inf'))
    output_q02 = K.clip(output_q02, min_value=-float('inf'), max_value=float('inf'))
    output_t01 = K.clip(output_t01, min_value=-float('inf'), max_value=float('inf'))

    # Epsilon prevents zero (loss goes inf)
    eps = K.epsilon()
    w, h = 200, 100
    tf_batch_size = tf.shape(optical_flow)[0]
    tf_height = tf.shape(optical_flow)[1]
    tf_width = tf.shape(optical_flow)[2]


    def custom_loss(y_true, y_pred):
        # output_t02 = y_pred
        y_pred = K.clip(y_pred, min_value=-float('inf'), max_value=float('inf'))

        # Quaternion
        qw01 = output_q01[:,0]
        qx01 = output_q01[:,1]
        qy01 = output_q01[:,2]
        qz01 = output_q01[:,0]
        q_norm = K.sqrt(K.square(qw01) + K.square(qx01) + K.square(qy01) + K.square(qz01) + eps)
        qw01, qx01, qy01, qz01 = qw01 / q_norm, -qx01 / q_norm, -qy01 / q_norm, -qz01 / q_norm

        FX = helper.get_3d_optical_flow(w, h, optical_flow)[0]
        FY = helper.get_3d_optical_flow(w, h, optical_flow)[1]
        FZ = helper.get_3d_optical_flow(w, h, optical_flow)[2]
        XYZ = helper.get_3d_optical_flow(w, h, optical_flow)[3]

        # 3D location derotated
        R11 = qw01 ** 2 + qx01 ** 2 - qy01 ** 2 - qz01 ** 2
        R12 = 2 * (qx01 * qy01 - qw01 * qz01)
        R13 = 2 * (qx01 * qz01 + qw01 * qy01)
        R21 = 2 * (qx01 * qy01 + qw01 * qz01)
        R22 = qw01 ** 2 - qx01 ** 2 + qy01 ** 2 - qz01 ** 2
        R23 = 2 * (qy01 * qz01 - qw01 * qx01)
        R31 = 2 * (qx01 * qz01 - qw01 * qy01)
        R32 = 2 * (qy01 * qz01 + qw01 * qx01)
        R33 = qw01 ** 2 - qx01 ** 2 - qy01 ** 2 + qz01 ** 2
        Rq = helper.get_tf_batch_matrix([
            [R11, R12, R13],
            [R21, R22, R23],
            [R31, R32, R33]])

        XYZ_derotated = K.reshape(tf.matmul(Rq, XYZ), [tf_batch_size, 3, tf_height, tf_width])
        X_derot_pred = XYZ_derotated[:, 0, :, :]
        Y_derot_pred = XYZ_derotated[:, 1, :, :]
        Z_derot_pred = XYZ_derotated[:, 2, :, :]

        disparity = K.sqrt((FX - X_derot_pred) ** 2 + (FY - Y_derot_pred) ** 2 + (FZ - Z_derot_pred) ** 2 + eps)

        tx01, ty01, tz01 = output_t01[:,0], output_t01[:,1], output_t01[:,2]
        t01 = K.sqrt(tx01**2 + ty01**2 + tz01**2 + eps)
        tx01, ty01, tz01 = -tx01/t01, -ty01/t01, -tz01/t01

        tx01 = K.reshape(K.tile(tx01, [h * w]), [tf_batch_size, h, w])
        ty01 = K.reshape(K.tile(ty01, [h * w]), [tf_batch_size, h, w])
        tz01 = K.reshape(K.tile(tz01, [h * w]), [tf_batch_size, h, w])

        omega = K.clip(
            X_derot_pred * tx01 + Y_derot_pred * ty01 + Z_derot_pred * tz01,
            min_value = -1.0 + eps,
            max_value = 1.0 - eps)
        omega = tf.acos(omega)

        # Fix translation(t0->t1) scale to 1.0
        fixed_translation_scale = 1.0
        depth = fixed_translation_scale * (K.abs(K.sin(omega + disparity)) + eps) / (K.sin(disparity) + eps) + eps
        #depth = tf.expand_dims(Depth, 3)

        # Gaussian filtered depth
        depth = tf.expand_dims(depth, 3)
        depth = tf.squeeze(helper.apply_gaussian_blur(depth, [7, 7], 2.0, 1), 3)

        # Median filtered depth
        #depth = tf.expand_dims(depth, 3)
        #depth = tf.squeeze(apply_median_blur(depth, 5))

        # Reconstruction loss with depth
        tx02, ty02, tz02 = y_pred[:,0], y_pred[:,1], y_pred[:,2]

        qw02 = output_q02[:,0]
        qx02 = output_q02[:,1]
        qy02 = output_q02[:,2]
        qz02 = output_q02[:,3]
        q02_norm = K.sqrt(qw02 ** 2 + qx02 ** 2 + qy02 ** 2 + qz02 ** 2 + eps)
        qw02 = qw02 / (q02_norm + eps)
        qx02 = qx02 / (q02_norm + eps)
        qy02 = qy02 / (q02_norm + eps)
        qz02 = qz02 / (q02_norm + eps)

        synthesized_image = reconstructor.rotate_and_translate(
            frame_t0,
            depth,
            qw02,
            qx02,
            qy02,
            qz02,
            tx02,
            ty02,
            tz02)
        synthesized_image = K.reshape(synthesized_image, [tf_batch_size, h, w, 3])

        # Weights
        W_kernel1 = K.tile(K.constant([1]), [int(w * h / 4) * tf_batch_size])
        W_kernel1 = K.reshape(W_kernel1, [tf_batch_size, int(w * h / 4)])
        W_kernel2 = K.tile(K.constant([1]), [int(w * h / 2) * tf_batch_size])
        W_kernel2 = K.reshape(W_kernel2, [tf_batch_size, int(w * h / 2)])

        W_kernel = K.concatenate([W_kernel1, W_kernel2], axis=-1)
        W_kernel = K.concatenate([W_kernel, W_kernel1], axis=-1)
        W_kernel = K.reshape(W_kernel, [tf_batch_size, h, w])

        Z = helper.get_3d_optical_flow(w, h, optical_flow)[4]
        W = K.sqrt(1 - Z * Z)
        W = W * W_kernel
        W = K.expand_dims(W, 3)
        W = K.tile(W, [1, 1, 1, 3])

        l1_loss = K.mean(K.abs(frame_t2 - synthesized_image) * W, axis=(1,2))
        return l1_loss

    return custom_loss


def get_multiscale_l1_loss(optical_flow, frame_t0, frame_t2, output_q01, output_q02, output_t01):

    # Optical flow 2ch
    optical_flow = K.clip(optical_flow, min_value=-float('inf'), max_value=float('inf'))
    frame_t0 = K.clip(frame_t0, min_value=-float('inf'), max_value=float('inf'))
    frame_t2 = K.clip(frame_t2, min_value=-float('inf'), max_value=float('inf'))
    output_q01 = K.clip(output_q01, min_value=-float('inf'), max_value=float('inf'))
    output_q02 = K.clip(output_q02, min_value=-float('inf'), max_value=float('inf'))
    output_t01 = K.clip(output_t01, min_value=-float('inf'), max_value=float('inf'))

    # Epsilon prevents zero (loss goes inf)
    eps = K.epsilon()
    w, h = 200, 100
    batch_size = tf.shape(optical_flow)[0]
    height = tf.shape(optical_flow)[1]
    width = tf.shape(optical_flow)[2]

    def custom_loss(y_true, y_pred):

        # output_t02 = y_pred
        y_pred = K.clip(y_pred, min_value=-float('inf'), max_value=float('inf'))

        # Quaternion
        qw01 = output_q01[:,0]
        qx01 = output_q01[:,1]
        qy01 = output_q01[:,2]
        qz01 = output_q01[:,0]
        q_norm = K.sqrt(K.square(qw01) + K.square(qx01) + K.square(qy01) + K.square(qz01) + eps)
        qw01, qx01, qy01, qz01 = qw01 / q_norm, -qx01 / q_norm, -qy01 / q_norm, -qz01 / q_norm

        FX = helper.get_3d_optical_flow(w, h, optical_flow)[0]
        FY = helper.get_3d_optical_flow(w, h, optical_flow)[1]
        FZ = helper.get_3d_optical_flow(w, h, optical_flow)[2]
        XYZ = helper.get_3d_optical_flow(w, h, optical_flow)[3]

        # 3D location derotated
        R11 = qw01 ** 2 + qx01 ** 2 - qy01 ** 2 - qz01 ** 2
        R12 = 2 * (qx01 * qy01 - qw01 * qz01)
        R13 = 2 * (qx01 * qz01 + qw01 * qy01)
        R21 = 2 * (qx01 * qy01 + qw01 * qz01)
        R22 = qw01 ** 2 - qx01 ** 2 + qy01 ** 2 - qz01 ** 2
        R23 = 2 * (qy01 * qz01 - qw01 * qx01)
        R31 = 2 * (qx01 * qz01 - qw01 * qy01)
        R32 = 2 * (qy01 * qz01 + qw01 * qx01)
        R33 = qw01 ** 2 - qx01 ** 2 - qy01 ** 2 + qz01 ** 2
        Rq = helper.get_tf_batch_matrix([
            [R11, R12, R13],
            [R21, R22, R23],
            [R31, R32, R33]])

        XYZ_derotated = K.reshape(tf.matmul(Rq, XYZ), [batch_size, 3, height, width])
        X_derot_pred = XYZ_derotated[:, 0, :, :]
        Y_derot_pred = XYZ_derotated[:, 1, :, :]
        Z_derot_pred = XYZ_derotated[:, 2, :, :]

        disparity = K.sqrt((FX - X_derot_pred) ** 2 + (FY - Y_derot_pred) ** 2 + (FZ - Z_derot_pred) ** 2 + eps)

        tx01, ty01, tz01 = output_t01[:,0], output_t01[:,1], output_t01[:,2]
        t01 = K.sqrt(tx01**2 + ty01**2 + tz01**2 + eps)
        tx01, ty01, tz01 = -tx01/t01, -ty01/t01, -tz01/t01

        tx01 = K.reshape(K.tile(tx01, [h * w]), [batch_size, h, w])
        ty01 = K.reshape(K.tile(ty01, [h * w]), [batch_size, h, w])
        tz01 = K.reshape(K.tile(tz01, [h * w]), [batch_size, h, w])

        omega = K.clip(
            X_derot_pred * tx01 + Y_derot_pred * ty01 + Z_derot_pred * tz01,
            min_value = -1.0 + eps,
            max_value = 1.0 - eps)
        omega = tf.acos(omega)

        # Fix translation(t0->t1) scale to 1.0
        fixed_translation_scale = 1.0
        depth = fixed_translation_scale * (K.abs(K.sin(omega + disparity)) + eps) /\
                (K.sin(disparity) + eps) + eps

        # Gaussian filtered depth
        depth = tf.expand_dims(depth, 3)
        depth = helper.apply_gaussian_blur(depth, [7, 7], 2.0, 1)

        # Reconstruction loss with depth
        tx02, ty02, tz02 = y_pred[:,0], y_pred[:,1], y_pred[:,2]
        qw02 = output_q02[:,0]
        qx02 = output_q02[:,1]
        qy02 = output_q02[:,2]
        qz02 = output_q02[:,0]
        q_norm = K.sqrt(K.square(qw02) + K.square(qx02) + K.square(qy02) + K.square(qz02) + eps)
        qw02, qx02, qy02, qz02 = qw02 / q_norm, -qx02 / q_norm, -qy02 / q_norm, -qz02 / q_norm

        # Resize for multi-scale
        frame_t0_1 = tf.image.resize_area(frame_t0, [int(h / (2**3)), int(w / (2**3))])  # 25x13
        frame_t0_2 = tf.image.resize_area(frame_t0, [int(h / (2**2)), int(w / (2**2))])  # 50x25
        frame_t0_3 = tf.image.resize_area(frame_t0, [int(h / (2**1)), int(w / (2**1))])  # 100x50
        frame_t0_4 = tf.image.resize_area(frame_t0, [int(h / (2**0)), int(w / (2**0))])  # 200x100

        frame_t2_1 = tf.image.resize_area(frame_t2, [int(h / (2**3)), int(w / (2**3))])
        frame_t2_2 = tf.image.resize_area(frame_t2, [int(h / (2**2)), int(w / (2**2))])
        frame_t2_3 = tf.image.resize_area(frame_t2, [int(h / (2**1)), int(w / (2**1))])
        frame_t2_4 = tf.image.resize_area(frame_t2, [int(h / (2**0)), int(w / (2**0))])

        depth_1 = tf.image.resize_nearest_neighbor(depth, [int(h / (2**3)), int(w / (2**3))])
        depth_2 = tf.image.resize_nearest_neighbor(depth, [int(h / (2**2)), int(w / (2**2))])
        depth_3 = tf.image.resize_nearest_neighbor(depth, [int(h / (2**1)), int(w / (2**1))])
        depth_4 = tf.image.resize_nearest_neighbor(depth, [int(h / (2**0)), int(w / (2**0))])

        # Synthetic images for multi-scale
        img_syn_1 = reconstructor.rotate_and_translate(
            frame_t0_1, tf.squeeze(depth_1, 3), qw02, qx02, qy02, qz02, tx02, ty02, tz02)
        img_syn_1 = tf.reshape(img_syn_1, [batch_size, int(h / (2**3)), int(w / (2**3)), 3])
        img_syn_2 = reconstructor.rotate_and_translate(
            frame_t0_2, tf.squeeze(depth_2, 3), qw02, qx02, qy02, qz02, tx02, ty02, tz02)
        img_syn_2 = tf.reshape(img_syn_2, [batch_size, int(h / (2**2)), int(w / (2**2)), 3])
        img_syn_3 = reconstructor.rotate_and_translate(
            frame_t0_3, tf.squeeze(depth_3, 3), qw02, qx02, qy02, qz02, tx02, ty02, tz02)
        img_syn_3 = tf.reshape(img_syn_3, [batch_size, int(h / (2**1)), int(w / (2**1)), 3])
        img_syn_4 = reconstructor.rotate_and_translate(
            frame_t0_4, tf.squeeze(depth_4, 3), qw02, qx02, qy02, qz02, tx02, ty02, tz02)
        img_syn_4 = tf.reshape(img_syn_4, [batch_size, int(h / (2**0)), int(w / (2**0)), 3])

        # Multi-distortion weights
        v_1 = K.tile(K.arange(0, int(h / (2**3)), step=1, dtype='float32'), [int(w / (2**3))])
        v_1 = K.transpose(K.reshape(v_1, [int(w / (2**3)), int(h / (2**3))]))
        v_2 = K.tile(K.arange(0, int(h / (2**2)), step=1, dtype='float32'), [int(w / (2**2))])
        v_2 = K.transpose(K.reshape(v_2, [int(w / (2**2)), int(h / (2**2))]))
        v_3 = K.tile(K.arange(0, int(h / (2**1)), step=1, dtype='float32'), [int(w / (2**1))])
        v_3 = K.transpose(K.reshape(v_3, [int(w / (2**1)), int(h / (2**1))]))
        v_4 = K.tile(K.arange(0, int(h / (2**0)), step=1, dtype='float32'), [int(w / (2**0))])
        v_4 = K.transpose(K.reshape(v_4, [int(w / (2**0)), int(h / (2**0))]))

        z_1 = K.cos(math.pi * v_1 / int(h / (2**3)))
        z_2 = K.cos(math.pi * v_2 / int(h / (2**2)))
        z_3 = K.cos(math.pi * v_3 / int(h / (2**1)))
        z_4 = K.cos(math.pi * v_4 / int(h / (2**0)))

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


def get_dummy_loss(y_true, y_pred):
    y_pred = K.clip(
        y_pred,
        min_value = -float('inf'),
        max_value = float('inf'))
    return y_pred