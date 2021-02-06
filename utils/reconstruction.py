import numpy as np
import tensorflow as tf
from utils.bilinear_sampler import bilinear_sample


def lat_long_grid(shape, epsilon=1.0e-12):
    return tf.meshgrid(tf.linspace(-np.pi, np.pi, shape[1]),
                       tf.linspace(-np.pi / 2.0 + epsilon, np.pi / 2.0 - epsilon, shape[0]))


# Convert Cartesian coordinates (x, y, z) to latitude (T) and longitude (S).
def xyz_to_lat_long(x, y, z):
    S = -tf.atan2(z, x)
    T = tf.atan2(tf.sqrt(x ** 2.0 + z ** 2.0), y)
    return S, T


# Convert latitude (T) and longitude (S) to Cartesian coordinates (x, y, z).
def lat_long_to_xyz(S, T):
    x = tf.cos(T) * tf.sin(S)
    y = tf.sin(T)
    z = tf.cos(T) * tf.cos(S)
    #c = tf.ones_like(T)
    return x, y, z


def lat_long_to_equirectangular_uv(S, T):
    # Convert latitude and longitude to UV coordinates
    # on an equirectangular plane.
    u = tf.mod(S / (2.0 * np.pi) - 0.25, 1.0)
    v = tf.mod(T / np.pi, 1.0)
    return u, v


# General rotation and translation function.
def rotate_and_translate(input_images, input_depth, qw, qx, qy, qz, tx, ty, tz, epsilon=1.0e-12):

    # Create constants.
    batch_size = tf.shape(input_images)[0]
    height = tf.shape(input_images)[1]
    width = tf.shape(input_images)[2]

    q_norm = tf.sqrt(qw**2 + qx**2 + qy**2 + qz**2 + epsilon)
    qw, qx, qy, qz = qw/q_norm, -qx/q_norm, -qz/q_norm, -qy/q_norm
    tx, ty, tz = -tx, -tz, -ty

    tx, ty, tz = tf.expand_dims(tx, 1), tf.expand_dims(ty, 1), tf.expand_dims(tz, 1)
    tx = tf.tile(tx, [1, width * height])
    ty = tf.tile(ty, [1, width * height])
    tz = tf.tile(tz, [1, width * height])

    # Function to parse Python lists as TensorFlow matrices.
    def tf_batch_matrix(matrix):
        return tf.transpose(tf.stack(matrix), [2, 0, 1])

    # Convert to Cartesian.
    S, T = lat_long_grid([height, width])
    x, y, z = lat_long_to_xyz(S, T)

    # Construct translation with depth (x/y/z)
    input_depth = tf.reshape(input_depth, [batch_size, height * width])
    input_depth = tf.tile(tf.expand_dims(input_depth, 1), [1, 3, 1])

    X = tf.tile(tf.expand_dims(tf.reshape(tf.stack([x, y, z]), [3, height * width]), 0), [batch_size, 1, 1])
    t = tf.reshape(tf.stack([tx, ty, tz], axis=1), [batch_size, 3, height * width])
    X = X * input_depth + t

    X_norm = tf.sqrt(X[:, 0, :] ** 2 + X[:, 1, :] ** 2 + X[:, 2, :] ** 2 + epsilon)
    X_norm = tf.tile(tf.expand_dims(X_norm, 1), [1, 3, 1])
    X /= X_norm

    # Construct rotation matrices (for inverse warp, quaternion qw/qx/qy/qz).
    R = tf_batch_matrix([
        [qw**2.0+qx**2.0-qy**2.0-qz**2.0, 2.0*(qx*qy-qw*qz), 2.0*(qx*qz+qw*qy)],
        [2.0*(qx*qy+qw*qz), qw**2.0-qx**2.0+qy**2.0-qz**2.0, 2.0*(qy*qz-qw*qx)],
        [2.0*(qx*qz-qw*qy), 2.0*(qy*qz+qw*qx), qw**2.0-qx**2.0-qy**2.0+qz**2.0]
    ])

    # Transform coordinates.
    X_transformed = tf.reshape(tf.matmul(R, X), [batch_size, 3, height, width])

    # Convert back to equirectangular UV.
    S_rotated, T_rotated = xyz_to_lat_long(X_transformed[:,0,:,:], X_transformed[:,1,:,:], X_transformed[:,2,:,:])
    u, v = lat_long_to_equirectangular_uv(S_rotated, T_rotated)
    image = bilinear_sample(input_images, x_t=tf.zeros_like(u[0]), y_t=tf.zeros_like(v[0]), x_offset=u, y_offset=1.0-v)
    return image


# General rotation function.
def rotate(input_images, qw, qx, qy, qz, epsilon=1.0e-12):

    # Create constants.
    batch_size = tf.shape(input_images)[0]
    height = tf.shape(input_images)[1]
    width = tf.shape(input_images)[2]

    qw = tf.tile(tf.constant([qw]), [batch_size])
    qx = tf.tile(tf.constant([qx]), [batch_size])
    qy = tf.tile(tf.constant([qy]), [batch_size])
    qz = tf.tile(tf.constant([qz]), [batch_size])

    q_norm = tf.sqrt(qw**2 + qx**2 + qy**2 + qz**2 + epsilon)
    qw, qx, qy, qz = qw/q_norm, -qx/q_norm, -qz/q_norm, -qy/q_norm

    # Function to parse Python lists as TensorFlow matrices.
    def tf_batch_matrix(matrix):
        return tf.transpose(tf.stack(matrix), [2, 0, 1])

    # Convert to Cartesian.
    S, T = lat_long_grid([height, width])
    x, y, z = lat_long_to_xyz(S, T)
    X = tf.tile(tf.expand_dims(tf.reshape(tf.stack([x, y, z]), [3, height * width]), 0), [batch_size, 1, 1])

    X_norm = tf.sqrt(X[:, 0, :] ** 2 + X[:, 1, :] ** 2 + X[:, 2, :] ** 2 + epsilon)
    X_norm = tf.tile(tf.expand_dims(X_norm, 1), [1, 3, 1])
    X /= X_norm

    # Construct rotation matrices (for inverse warp, quaternion qw/qx/qy/qz).
    R = tf_batch_matrix([
        [qw**2.0+qx**2.0-qy**2.0-qz**2.0, 2.0*(qx*qy-qw*qz), 2.0*(qx*qz+qw*qy)],
        [2.0*(qx*qy+qw*qz), qw**2.0-qx**2.0+qy**2.0-qz**2.0, 2.0*(qy*qz-qw*qx)],
        [2.0*(qx*qz-qw*qy), 2.0*(qy*qz+qw*qx), qw**2.0-qx**2.0-qy**2.0+qz**2.0]
    ])

    # Transform coordinates.
    X_transformed = tf.reshape(tf.matmul(R, X), [batch_size, 3, height, width])

    # Convert back to equirectangular UV.
    S_rotated, T_rotated = xyz_to_lat_long(X_transformed[:,0,:,:], X_transformed[:,1,:,:], X_transformed[:,2,:,:])
    u, v = lat_long_to_equirectangular_uv(S_rotated, T_rotated)
    image = bilinear_sample(input_images, x_t=tf.zeros_like(u[0]), y_t=tf.zeros_like(v[0]), x_offset=u, y_offset=1.0-v)
    return image


def rotate_test():
    # Load equirectangular image.
    filename = 'data/frame00001'

    height = 100
    equirectangular_image = read_image(filename + ".png", [height, 2*height])

    # Transformation
    qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0

    # Axis arrangement
    qw, qx, qy, qz = qw, -qx, -qz, -qy  # Blender script coordinates
    rw, rx, ry, rz = tf.stack([qw]), tf.stack([qx]), tf.stack([qy]), tf.stack([qz])
    equirectangular_image = tf.tile(equirectangular_image, [1, 1, 1, 1])

    # Rotate image.
    rotated_images = rotate(equirectangular_image, rw, rx, ry, rz)
    session = tf.compat.v1.Session()

    image_data = session.run(encode_images(rotated_images, 1))
    for index in range(1):
        write_image(image_data[index], filename + "_result.png".format(index))


def read_image(image_path, shape):
    if image_path.lower().endswith("png"):
        image = tf.image.decode_png(tf.io.read_file(image_path))
    else:
        image = tf.image.decode_jpeg(tf.io.read_file(image_path))
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, shape, tf.image.ResizeMethod.AREA)
    return tf.expand_dims(image, 0)


def read_depth_png(depth_path, shape):
    if depth_path.lower().endswith("png"):
        depth = tf.image.decode_png(tf.io.read_file(depth_path))
    else:
        depth = tf.image.decode_jpeg(tf.io.read_file(depth_path))
    depth = tf.image.resize(depth, shape, tf.image.ResizeMethod.AREA)
    return tf.expand_dims(depth, 0)


def read_depth_npy(depth_path, height, width, dim):
    depth = np.load(depth_path)
    depth = tf.convert_to_tensor(depth, dtype=tf.float32)
    depth = tf.reshape(depth, [height, width, dim])
    return tf.expand_dims(depth, 0)


def write_image(image_data, filename):
    with open(filename, "wb") as image_file:
        image_file.write(image_data)


def encode_image(image, type="jpg", index=0):
    quantized_image = tf.image.convert_image_dtype(image[index, :, :, :], tf.uint8)
    if type == "png":
        return tf.image.encode_png(quantized_image)
    else:
        return tf.image.encode_jpeg(quantized_image)


def encode_images(images, batch_size, type="png"):
    return [encode_image(images, type, index) for index in range(batch_size)]


if __name__ == "__main__":
    rotate_test()
