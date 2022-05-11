"""CerraNet model for Keras.

# Reference

- ["CerraNet: a deep convolutional neural network for classifying 
    land use and land cover on Cerrado biome tocantinense"](
    https://drive.google.com/file/d/1JnN52C8yZKwN-5XA6qSiCsCygh1-0vvZ/view)

"""
import os

import keras
from keras import backend, layers, models
if keras.__version__ < '2.4.0':
    from keras import utils as keras_utils
else:
    from keras.utils import data_utils as keras_utils

from keras_applications.imagenet_utils import _obtain_input_shape

WEIGHTS_PATH = ('file://' + 
                os.path.dirname(os.path.abspath(__file__)) + '/'
                'models/'
                'cerranet_weights_tf_dim_ordering_tf_kernels.h5')
WEIGHTS_PATH_NO_TOP = ('file://' + 
                       os.path.dirname(os.path.abspath(__file__)) + '/'
                       'models/'
                       'cerranet_weights_tf_dim_ordering_tf_kernels_notop.h5')


def CerraNet(include_top=True,
             weights='cerranet',
             input_tensor=None,
             input_shape=None,
             pooling=None,
             classes=4,
             **kwargs):
    """Instantiates the CerraNet architecture.

    Optionally loads weights pre-trained on Sports1M.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.

    # Arguments
        include_top: whether to include the 3 fully-connected
            layers at the top of the network.
        weights: one of `None` (random initialization),
              'cerranet' (pre-training on CerraNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor
            (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(256, 256, 3)`
            (with `channels_last` data format)
            or `(3, 256, 256)` (with `channels_first` data format).
            It should have exactly 3 input channels,
            and width and height should be no smaller than 128.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    if not (weights in {'cerranet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `cerranet` '
                         '(pre-training on CerraNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'cerranet' and include_top and classes != 4:
        raise ValueError('If using `weights` as `"cerranet"` with `include_top`'
                         ' as true, `classes` should be 4')
    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=256,
                                      min_size=128,
                                      data_format=backend.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    # Block 1
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      name='conv1')(img_input)
    x = layers.AveragePooling2D((2, 2), name='pool1')(x)
    x = layers.Dropout(0.15)(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      name='conv2')(x)
    x = layers.AveragePooling2D((2, 2), name='pool2')(x)
    x = layers.Dropout(0.15)(x)

    # Block 3
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      name='conv3')(x)
    x = layers.AveragePooling2D((2, 2), name='pool3')(x)
    x = layers.Dropout(0.15)(x)

    # Block 4
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      name='conv4')(x)
    x = layers.AveragePooling2D((2, 2), name='pool4')(x)
    x = layers.Dropout(0.15)(x)

    # Block 5
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      name='conv5')(x)
    x = layers.AveragePooling2D((2, 2), name='pool5')(x)
    x = layers.Dropout(0.15)(x)

    # Block 6
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      name='conv6')(x)
    x = layers.AveragePooling2D((2, 2), name='pool6')(x)
    x = layers.Dropout(0.15)(x)

    if include_top:
        # Classification block
        x = layers.Flatten(name='flatten')(x)
        x = layers.Dense(256, activation='relu', name='fc7')(x)
        x = layers.Dropout(0.15)(x)
        x = layers.Dense(128, activation='relu', name='fc8')(x)
        x = layers.Dropout(0.15)(x)
        x = layers.Dense(classes, activation='softmax', name='fc9')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = models.Model(inputs, x, name='cerranet')

    # Load weights.
    if weights == 'cerranet':
        if include_top:
            weights_path = keras_utils.get_file(
                'cerranet_weights_tf_dim_ordering_tf_kernels.h5',
                WEIGHTS_PATH,
                cache_subdir='models',
                file_hash='7ff1f7be9905829ef85e0167313700d6')
        else:
            weights_path = keras_utils.get_file(
                'cerranet_weights_tf_dim_ordering_tf_kernels_notop.h5',
                WEIGHTS_PATH_NO_TOP,
                cache_subdir='models',
                file_hash='eb32da6e2d5225ff2c0e85e256b1ecdd')
        model.load_weights(weights_path)
        if backend.backend() == 'theano':
            keras_utils.convert_all_kernels_in_model(model)
    elif weights is not None:
        model.load_weights(weights)

    return model
