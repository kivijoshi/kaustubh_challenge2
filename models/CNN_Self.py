#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 17:40:14 2016

@author: kaustubh
"""
from keras.models import model_from_json, Sequential
from keras.layers import Activation, Dense, Dropout, Flatten, Lambda, LSTM, ELU
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.wrappers import TimeDistributed
from keras.layers.core import Reshape,Permute
from keras.optimizers import SGD
from keras.preprocessing.image import load_img,img_to_array
import argparse
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
from pprint import pprint
from prettytable import PrettyTable

def CNN_Self(volumesPerBatch, timesteps, cameraFormat=(3, 224, 224), verbosity=0):
  """
  Build and return a CNN + LSTM model; details in the comments.

  The model expects batch_input_shape =
  (volumes per batch, timesteps per volume, (camera format 3-tuple))

  A "volume" is a video frame data struct extended in the time dimension.

  Args:
    volumesPerBatch: (int) batch size / timesteps
    timesteps: (int) Number of timesteps per volume.
    cameraFormat: (3-tuple) Ints to specify the input dimensions (color
        channels, height, width).
    verbosity: (int) Print model config.
  Returns:
    A compiled Keras model.
  """
  print "Building model..."
  ch, row, col = cameraFormat

  model = Sequential()

  if timesteps == 1:
    raise ValueError("Not supported w/ TimeDistributed layers")

  # Use a lambda layer to normalize the input data
  # It's necessary to specify batch_input_shape in the first layer in order to
  # have stateful recurrent layers later
#  model.add(Lambda(
#      lambda x: x/127.5 - 1.,),batch_input_shape=(volumesPerBatch,row, col,ch),)

  # For CNN layers, weights are initialized with Gaussian scaled by fan-in and
  # activation is via ReLU units; this is current best practice (He et al., 2014)

  # Several convolutional layers, each followed by ELU activation
  # 8x8 convolution (kernel) with 4x4 stride over 16 output filters
#  model.add(Lambda(
#      lambda x: x/127.5 - 1.,
#      input_shape=(row, col,ch),
#      output_shape=(row, col,ch))
#  )

  # Several convolutional layers, each followed by ELU activation
  # 8x8 convolution (kernel) with 4x4 stride over 16 output filters
  model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same",input_shape=(row, col,ch)))
  model.add(Activation("relu"))
  # 5x5 convolution (kernel) with 2x2 stride over 32 output filters
  model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
  model.add(Activation("relu"))
  # 5x5 convolution (kernel) with 2x2 stride over 64 output filters
  model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
  # Flatten the input to the next layer
  model.add(Flatten())
  # Apply dropout to reduce overfitting
  model.add(Dropout(.2))
  model.add(Activation("relu"))
  # Fully connected layer
  
  model.add(Dense(5012))
  model.add(Dropout(.5))
  model.add(Activation("relu"))
  
  model.add(Dense(2056))
  # More dropout
  model.add(Dropout(.5))
  model.add(Activation("relu"))

  model.add(Dense(1024))
  model.add(Dropout(.5))
  model.add(Activation("relu"))
  # More dropout
  
  model.add(Dense(512))
  model.add(Dropout(.5))
  model.add(Activation("relu"))

  model.add(Dense(10))
  model.add(Dropout(.5))
  model.add(Activation("relu"))
  
  
  model.add(Dense(1))

  # Adam optimizer is a standard, efficient SGD optimization method
  # Loss function is mean squared error, standard for regression problems
  backprop = SGD(lr=0.001, momentum=0.0, decay=0.0, nesterov=False)
  model.compile(optimizer=backprop, loss="mse")

  if verbosity:
    printTemplate = PrettyTable(["Layer", "Input Shape", "Output Shape"])
    printTemplate.align = "l"
    printTemplate.header_style = "upper"
    for layer in model.layers:
      printTemplate.add_row([layer.name, layer.input_shape, layer.output_shape])
    print printTemplate

  if verbosity > 1:
    config = model.get_config()
    for layerSpecs in config:
      pprint(layerSpecs)

  return model
