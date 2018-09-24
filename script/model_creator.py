import sys

import keras
import resnet as kr
from keras.layers import *
from keras.models import Model
from keras.utils import plot_model

from global_parameters import *


def model_creator(model_type, show_summary=False):
    """
        Generate the model.
        Available two architcture
            -middle: predicts user rel pose
            -controller: predicts twist message
    Args:
        model_type: type of architecture created
        show_summary: if true, show keras model summary in console

    Returns:
        model ready for train
    """
    if model_type == "v1":
        input_shape = (3, image_height, image_width)
        model_input = Input((image_height, image_width, 3), name="image_input")
        blocks = [1, 1, 1]
        resnet_custom = build_resnet(input_shape, kr.basic_block, blocks, show_summary)
        rn_out = resnet_custom(model_input)
        f_1 = (Flatten())(rn_out)
        d_1 = (Dense(256, activation='relu', name="1_dense"))(f_1)
        d_2 = (Dense(128, activation='relu', name="2_dense"))(d_1)
        y_1 = (Dense(1, activation='linear', name=output_names_v1[0]))(d_2)
        y_2 = (Dense(1, activation='linear', name=output_names_v1[1]))(d_2)
        y_3 = (Dense(1, activation='linear', name=output_names_v1[2]))(d_2)
        y_4 = (Dense(1, activation='linear', name=output_names_v1[3]))(d_2)
        model = Model(inputs=model_input, outputs=[y_1, y_2, y_3, y_4])
        model.compile(loss='mean_absolute_error',
                      optimizer='adam',
                      metrics=['mse'])
        plot_model(model, to_file='../model_v1_2.png', show_shapes=True)

        if show_summary:
            model.summary()

    elif model_type == "v2":
        input_shape = (3, image_height, image_width)
        model_input = Input((image_height, image_width, 3), name="image_input")
        blocks = [1, 1, 1]
        resnet_custom = build_resnet(input_shape, kr.basic_block, blocks, show_summary)
        rn_out = resnet_custom(model_input)
        f_1 = (Flatten())(rn_out)
        auxiliary_input = Input(shape=(2,), name='odom_vel_input')
        x = keras.layers.concatenate([f_1, auxiliary_input])
        d_1 = (Dense(256, activation='relu', name="1_dense"))(x)
        d_2 = (Dense(128, activation='relu', name="2_dense"))(d_1)
        y_1 = (Dense(1, activation='linear', name=output_names_v2[0]))(d_2)
        y_2 = (Dense(1, activation='linear', name=output_names_v2[1]))(d_2)
        y_3 = (Dense(1, activation='linear', name=output_names_v2[2]))(d_2)
        y_4 = (Dense(1, activation='linear', name=output_names_v2[3]))(d_2)
        model = Model(inputs=[model_input, auxiliary_input], outputs=[y_1, y_2, y_3, y_4])
        model.compile(loss='mean_absolute_error',
                      optimizer='adam',
                      metrics=['mse'])
        plot_model(model, to_file='../model_v2_2.png', show_shapes=True)

        if show_summary:
            model.summary()

    elif model_type == "v3":
        model_input = Input((4,), name="rel_pose_input")
        auxiliary_input = Input(shape=(2,), name='odom_vel_input')
        x = keras.layers.concatenate([model_input, auxiliary_input])
        d_1 = (Dense(256, activation='relu', name="1_dense"))(x)
        d_2 = (Dense(128, activation='relu', name="2_dense"))(d_1)
        y_1 = (Dense(1, activation='linear', name=output_names_v2[0]))(d_2)
        y_2 = (Dense(1, activation='linear', name=output_names_v2[1]))(d_2)
        y_3 = (Dense(1, activation='linear', name=output_names_v2[2]))(d_2)
        y_4 = (Dense(1, activation='linear', name=output_names_v2[3]))(d_2)
        model = Model(inputs=[model_input, auxiliary_input], outputs=[y_1, y_2, y_3, y_4])
        model.compile(loss='mean_absolute_error',
                      optimizer='adam',
                      metrics=['mse'])
        plot_model(model, to_file='../model_v3_2.png', show_shapes=True)

        if show_summary:
            model.summary()

    else:
        print("ERROR: Model flag passed is not correct in model creator: " + model_type)
        sys.exit(1)
    return model


def build_resnet(input_shape, block_fn, repetitions, plot_model_f):
    """Builds a custom ResNet like architecture.
    Args:
        input_shape: The input shape in the form (nb_channels, nb_rows, nb_cols)
        num_outputs: The number of outputs at final softmax layer
        block_fn: The block function to use. This is either `basic_block` or `bottleneck`.
            The original paper used basic_block for layers < 50
        repetitions: Number of repetitions of various block units.
            At each block unit, the number of filters are doubled and the input size is halved
    Returns:
        The resnet part of the model.
    """
    kr._handle_dim_ordering()
    if len(input_shape) != 3:
        raise Exception("Input shape should be a tuple (nb_channels, nb_rows, nb_cols)")

    # Permute dimension order if necessary
    if K.image_dim_ordering() == 'tf':
        input_shape = (input_shape[1], input_shape[2], input_shape[0])

    # Load function from str if needed.
    block_fn = kr._get_block(block_fn)

    input = Input(shape=input_shape)
    conv1 = kr._conv_bn_relu(filters=64, kernel_size=(7, 7), strides=(2, 2))(input)
    pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(conv1)

    block = pool1
    filters = 64
    for i, r in enumerate(repetitions):
        block = kr._residual_block(block_fn, filters=filters, repetitions=r, is_first_layer=(i == 0))(block)
        filters *= 2

    # Last activation
    block = kr._bn_relu(block)

    # Classifier block
    block_shape = K.int_shape(block)
    pool2 = AveragePooling2D(pool_size=(block_shape[kr.ROW_AXIS], block_shape[kr.COL_AXIS]),
                             strides=(1, 1))(block)

    model = Model(inputs=input, outputs=pool2)
    # plot_model(model, to_file='../model_resnet_inner.png', show_shapes=True)
    return model


def data_augmentor(sample, target, model_type):
    """
        recieves frame and targets with p=50% flip vertically
    Args:
        target:  target for CNN
        model_type:
        sample:
    Returns:
        datapoints and targets eventually flipped

    """
    if model_type == "v1":
        frame = sample
        if np.random.choice([True, False]):
            frame = np.fliplr(frame)  # IMG
            target[1] = -target[1]  # Y
            target[3] = -target[3]  # Relative YAW
        return frame, target
    elif model_type == "v2":
        frame = sample[0]
        vel = sample[1]
        x_2 = np.asarray([vel[0], vel[1]])
        if np.random.choice([True, False]):
            frame = np.fliplr(frame)  # IMG
            target[1] = -target[1]  # y_lin
            target[3] = -target[3]  # z_ang
            x_2 = np.asarray([vel[0], -vel[1]])
        return frame, x_2, target
    elif model_type == "v3":
        rel_pose = sample[0]
        vel = sample[1]
        x_2 = np.asarray([vel[0], vel[1]])
        if np.random.choice([True, False]):
            rel_pose[1] = -rel_pose[1]
            rel_pose[3] = -rel_pose[3]
            target[1] = -target[1]  # y_lin
            target[3] = -target[3]  # z_ang
            x_2 = np.asarray([vel[0], -vel[1]])
        return rel_pose, x_2, target
    else:
        print("ERROR: Model flag in data augmentor is not correct: " + model_type)
        sys.exit(1)


def generator(samples, targets, batch_size, model_type):
    """
        Genereator of minibatches of size batch_size
    Args:
        samples: sample array
        targets: targets array
        batch_size: batch size
        model_type: type of architecture created
    Yields:
        batch of samples and array of batch of targets
    """

    if model_type == "v1":
        while True:
            indexes = np.random.choice(np.arange(0, samples.shape[0]), batch_size)
            batch_samples = samples[indexes]
            batch_targets = targets[indexes]
            for i in range(0, batch_samples.shape[0]):
                batch_samples[i], batch_targets[i] = data_augmentor(batch_samples[i], batch_targets[i], model_type)
            yield batch_samples, [batch_targets[:, 0], batch_targets[:, 1], batch_targets[:, 2], batch_targets[:, 3]]
    elif model_type == "v2":
        while True:
            samples_1 = samples[0]
            samples_2 = samples[1]
            indexes = np.random.choice(np.arange(0, samples_1.shape[0]), batch_size)
            batch_samples_1 = samples_1[indexes]
            batch_samples_2 = samples_2[indexes]
            batch_targets = targets[indexes]
            for i in range(0, batch_samples_1.shape[0]):
                batch_samples_1[i], batch_samples_2[i], batch_targets[i] = data_augmentor([batch_samples_1[i], batch_samples_2[i]], batch_targets[i], model_type)
            yield [batch_samples_1, np.asarray(batch_samples_2)], [batch_targets[:, 0], batch_targets[:, 1], batch_targets[:, 2], batch_targets[:, 3]]
    elif model_type == "v3":
        while True:
            samples_1 = samples[0]
            samples_2 = samples[1]
            indexes = np.random.choice(np.arange(0, samples_1.shape[0]), batch_size)
            batch_samples_1 = samples_1[indexes]
            batch_samples_2 = samples_2[indexes]
            batch_targets = targets[indexes]
            for i in range(0, batch_samples_1.shape[0]):
                batch_samples_1[i], batch_samples_2[i], batch_targets[i] = data_augmentor([batch_samples_1[i], batch_samples_2[i]], batch_targets[i], model_type)
            yield [batch_samples_1, np.asarray(batch_samples_2)], [batch_targets[:, 0], batch_targets[:, 1], batch_targets[:, 2], batch_targets[:, 3]]
    else:
        print("ERROR: Model flag in generator is not correct: " + model_type)
        sys.exit(1)

