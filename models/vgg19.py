# -*- coding: utf-8 -*-
'''VGG19 model for Keras.

# Reference:

- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)

'''
from __future__ import print_function

import numpy as np
import warnings

from keras.models import Model
from keras.layers import Flatten, Dense, Input
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import  Dropout,LSTM,Activation
from keras.preprocessing import image
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
from keras.layers.wrappers import TimeDistributed
from keras.layers.core import Reshape,Permute
from pprint import pprint
from prettytable import PrettyTable
from keras.models import model_from_json, Sequential
from keras import backend as K
from imagenet_utils import decode_predictions, preprocess_input


TH_WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_th_dim_ordering_th_kernels.h5'
TF_WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels.h5'
TH_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_th_dim_ordering_th_kernels_notop.h5'
TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'


def VGG19(volumesPerBatch, timesteps, cameraFormat=(3, 224, 224), verbosity=0,include_top=True,input_tensor=None, weights=None):
    '''Instantiate the VGG19 architecture,
    optionally loading weights pre-trained
    on ImageNet. Note that when using TensorFlow,
    for best performance you should set
    `image_dim_ordering="tf"` in your Keras config
    at ~/.keras/keras.json.

    The model and the weights are compatible with both
    TensorFlow and Theano. The dimension ordering
    convention used by the model is the one
    specified in your Keras config file.

    # Arguments
        include_top: whether to include the 3 fully-connected
            layers at the top of the network.
        weights: one of `None` (random initialization)
            or "imagenet" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.

    # Returns
        A Keras model instance.
    '''
    print("Building model...")
    ch, row, col = cameraFormat
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')
    # Determine proper input shape
    if K.image_dim_ordering() == 'th':
        if include_top:
            input_shape = (3, 224, 224)
        else:
            input_shape = (3, None, None)
    else:
        if include_top:
            input_shape = (224, 224, 3)
        else:
            input_shape = (None, None, 3)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor)
        else:
            img_input = input_tensor
    model = Sequential()
    # Block 1
    model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv1',input_shape=(row, col, ch)))
    model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

    # Block 2
    model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv1'))
    model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

    # Block 3
    model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv1'))
    model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv2'))
    model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv3'))
    model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv4'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

#    # Block 4
    model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv1'))
    model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv2'))
    model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv3'))
    model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv4'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))

    # Block 5
    model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block5_conv1'))
    model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block5_conv2'))
    model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block5_conv3'))
    model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block5_conv4'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))

    if include_top:
        # Classification block
        model.add(Flatten())
        #model.add(Dropout(.2))
        #print(model.output_shape)
        #model.add(Reshape(target_shape=(timesteps,3136), name='reshape'))
        model.add(Dense(2048, name='fc1'))
        model.add(Dropout(.4))
        model.add(Activation("relu"))
        model.add(Dense(1024,name='fc2'))
        model.add(Dropout(.3))
        model.add(Activation("relu"))
        model.add(Dense(512, name='fc3'))
        model.add(Dropout(.2))
        model.add(Activation("relu"))
        model.add(Dense(1))

    # Create model

    model.compile(optimizer="adam", loss="mse")

    if verbosity:
        printTemplate = PrettyTable(["Layer", "Input Shape", "Output Shape"])
        printTemplate.align = "l"
        printTemplate.header_style = "upper"
        for layer in model.layers:
          printTemplate.add_row([layer.name, layer.input_shape, layer.output_shape])
        print(printTemplate)
    
    if verbosity > 1:
        config = model.get_config()
        for layerSpecs in config:
          pprint(layerSpecs)
    return model


#if __name__ == '__main__':
#    model = VGG19(include_top=True, weights='imagenet')
#
#    img_path = 'cat.jpg'
#    img = image.load_img(img_path, target_size=(224, 224))
#    x = image.img_to_array(img)
#    x = np.expand_dims(x, axis=0)
#    x = preprocess_input(x)
#    print('Input image shape:', x.shape)
#
#    preds = model.predict(x)
#    print('Predicted:', decode_predictions(preds))
