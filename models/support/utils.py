"""
Utility functions for the models.
"""
import cv2
import json
import logging
import numpy as np
import os
import csv

from keras.callbacks import Callback






def robustFramesGenerator(truthData,Images, batchSize,
                            timesteps=10, speedVisuals=False, verbosity=0):
    """ Yields X and Y data (in batches). Each yield is the frame data extended
    in the time dimension, or a "volume", where there are batchSize / timesteps
    volumes per batch.

    Args:
      truthData: (list) Truth data, as a list of [time in seconds, value] items.
          We assume that items correspond to successive video frames.
      batchSize: (int) Number of consecutive frames per batch.
      timesteps: (int) Number of timesteps to accumulate in a volume.
      speedVisuals: (bool) Draw visuals over the video.
      verbosity: (int) Print detailed info or not.
    Yields:
      Two-tuple representing a batch of input video frames and target values.
    """
    if timesteps < 1:
      raise ValueError("Need more than 0 timesteps.")
    if batchSize%timesteps:
      raise ValueError(
          "Batch size should be divisible by timesteps so we get an equal "
          "number of frames in each portion of the batch.")

    # A "volume" is a video frame data struct extended in the time dimension.
    volumesPerBatch = batchSize / timesteps
    height = 480
    width = 640
    if timesteps > 1:
      # For recurrent architectures
      X = np.zeros((volumesPerBatch, timesteps, 3, height, width), dtype="uint8")
      Y = np.zeros((volumesPerBatch, timesteps, 1), dtype="float32")
    else:
      # For static models (no time dimension)
      X = np.zeros((batchSize, 3, height, width), dtype="uint8")
      Y = np.zeros((batchSize, 1), dtype="float32")

    if verbosity:
      print "Data shapes from the generator:"
      print "  X =", X.shape
      print "  Y =", Y.shape

    # Loop through video and accumulate data (over time for each batch)
#    batchCount = 0
#    volumeIndex = -1
#    for frameIdx, value in enumerate(truthData):
#      ret, frame = self.videoSource.read()
#      if ret is False: continue
#
#      # Update counters so we know where to allocate this frame in the data structs
#      timeIndex = frameIdx%timesteps
#      if timeIndex == 0:
#        volumeIndex += 1
#
#      # Populate data structs; in the frame we roll the RGB dimension to the front
#      if timesteps > 1:
#        X[volumeIndex][timeIndex, :, :, :] = np.rollaxis(frame, 2)
#        Y[volumeIndex][timeIndex] = value
#      else:
#        batchIndex = frameIdx%batchSize
#        X[batchIndex] = np.rollaxis(frame, 2)
#        Y[batchIndex] = value
#
#      if volumeIndex==volumesPerBatch-1 and timeIndex==timesteps-1:
#        # End of this batch
#        if verbosity:
#          print "Now yielding batch", batchCount
#        batchCount += 1
#        volumeIndex = 0
#        yield X, Y


def prepTruthData(dataPath, normalizeData=False):
  """
  Get and preprocess the ground truth drive speeds data.

  Args:
    dataPath: (str) Path to JSON of ground truths.
    numFrames: (int) Number of timesteps to interpolate the data to.
    normalizeData: (bool) Normalize the data to [0,1].
  Returns:
    (list) Linearly interpolated truth values, one for each timestep.
  """
  with open(dataPath, "rb") as steeringfile:
    driveData = np.loadtxt(steeringfile,delimiter=",",skiprows = 1)

  # Prep data: make sure it's in order, and use relative position (b/c seconds
  # values may be incorrect)
  #driveData.sort(key = lambda x: x[0])
  times = driveData[:,1]
  steeringAngles = driveData[:,2]
  
  # Linearly interpolate the data to the number of video frames
  return times,steeringAngles



def normalize(vector):
  nump = np.array(vector)
  return (nump - nump.min()) / (nump.max() - nump.min())



def standardize(array):
  nump = np.array(array)
  return (nump - nump.mean()) / nump.std()



class LossHistory(Callback):
  """ Helper class for useful logging info; set logger to DEBUG.
  """
  def on_train_begin(self, logs={}):
    self.losses = []

  def on_epoch_end(self, epoch, logs={}):
    self.losses.append(logs.get("val_loss"))
    logging.debug("-----------------------------------------------------------")
    logging.debug("Epoch {} \nValidation loss = {} \nAccuracy = {}".format(
        epoch, logs.get("val_loss"), logs.get("val_acc")))
    logging.debug("-----------------------------------------------------------")

  def on_batch_end(self,batch,logs={}):
    logging.debug("Batch {} -- loss = {}, accuracy = {}".format(
        batch, logs.get("loss"), logs.get("acc")))


