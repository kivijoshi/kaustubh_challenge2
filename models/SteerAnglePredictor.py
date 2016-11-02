#!/usr/bin/env python
"""
Auther: Kaustubh Joshi

How To Use : 

 

This work is inspired by the 
1. https://github.com/BoltzmannBrain/self-driving
2. https://github.com/fchollet/deep-learning-models
"""

from keras.models import model_from_json, Sequential
from keras.layers import Activation, Dense, Dropout, Flatten, Lambda, LSTM
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.wrappers import TimeDistributed
from keras.layers.core import Reshape,Permute
from keras.preprocessing.image import load_img,img_to_array
import argparse
import json
import logging
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import os
from pprint import pprint
from prettytable import PrettyTable
from vgg19 import VGG19
from New_Model import NativeModel
from CNN_Self import CNN_Self
from New_Model_Original import NativeModel2

from support.utils import prepTruthData, LossHistory,robustFramesGenerator
import tensorflow as tf
tf.python.control_flow_ops = tf

_DEFAULT_MODEL_DIR = "./outputs/steer_model"

# Model architecture is based on video defaults:
_VIDEO_CHANNELS = 3
_VIDEO_HEIGHT = 224
_VIDEO_WIDTH = 224

logging.basicConfig(level=logging.INFO)

def DataPreparation(ImagePath, ImageSeqFile, dataPath):
  """ Setup the data -- create VideoStream object and preprocess the truth data.

  Args:
    ImagePath: Path to image foder
    dataPath: Path to the truth data csv
  Returns:
    images : full path of the images
    truthData : steering wheel ground truth data interpolated to the timestamp of the images
  """

  times, steeringAngles = prepTruthData(dataPath)
  with open(ImageSeqFile, "rb") as imageSequenceFile:
    imageSequence = np.genfromtxt(imageSequenceFile,dtype = None, delimiter=",",skip_header = 1)
  
  checkOnTimes = []
  images = []
  for line in imageSequence:
          checkOnTimes.append(line[1]) 
          images.append(ImagePath+line[5])
         
  truthData = np.interp(checkOnTimes,times ,steeringAngles)
  truthData = np.asarray(truthData)
  return images, truthData

#def DataPreparation_LSTM(ImagePath, ImageSeqFile, dataPath,timesteps):
#  """ Setup the data -- create VideoStream object and preprocess the truth data.
#
#  Args:
#    ImagePath: Path to image foder
#    dataPath: Path to the truth data csv
#  Returns:
#    images : full path of the images
#    truthData : steering wheel ground truth data interpolated to the timestamp of the images
#  """
#  # Open video stream
#  #videoStream = initVideoStream(videoPath)
#  #videoInfo = videoStream.getVideoInfo()
#
#  # Get and prep training data
#  times, steeringAngles = prepTruthData(dataPath)
#  with open(ImageSeqFile, "rb") as imageSequenceFile:
#    imageSequence = np.genfromtxt(imageSequenceFile,dtype = None, delimiter=",",skip_header = 1)
#  
#  checkOnTimes = []
#  images = []
#  for line in imageSequence:
#      if(line[4] == "center_camera"):
#          checkOnTimes.append(line[1]) 
#          images.append(ImagePath+line[5])
#         
#  truthData = np.interp(checkOnTimes,times ,steeringAngles)
#  truthData = np.asarray(truthData)
#  
#  images_c = []
#  truthData_c = []
#  for i in range(len(images)-(timesteps-1)):
#      for j in range(timesteps):
#          images_c.append(images[i+j])
#          truthData_c.append(truthData[i+(timesteps-1)])
#          
#  return images_c, truthData_c


def getDataBatch(EndPoint,Start, DataBatchSize):
    
    """
    This shall not be confused with batch fetching. This function will keep on providing data to train and test function 
    periodically. This has been done to optimize the memory consumption
    
    
    """
    volumesPerBatch = int(args.batchSize)
    images, truthData = DataPreparation(args.ImagesPath,args.ImageSeqFile, args.dataPath)
    X = []
    Y = []
    if(Start+DataBatchSize<EndPoint):
        End = Start+DataBatchSize
    else:
        End = EndPoint
        
    #cnt = 0        
    for i in range(Start,End):
      img = load_img(images[i],target_size=(224, 224))
      x = img_to_array(img, dim_ordering='tf')
      X.append(x)
      Y.append(truthData[i])
#      cnt = cnt + 1
#      if(cnt == args.timesteps):
#          
#          cnt = 0

      
    X = np.asarray(X)
    X = np.reshape(X,(len(X),224,224,3))
    Y = np.asarray(Y)
    Y = np.reshape(Y,(X.shape[0],1))
    
    return X,Y,End
    

    
    
def _runTrain(args,Images, truthData, TrainCnt,TestCnt):
  """
  Builds or loads a model (with global defaults for frame specs), trains it on
  the specified video stream, and saves the model weights and architecture.

  Args:
    (See the script's command line arguments.)
    videoStream: (VideoStream) The generator is used to yield X and Y training data batches.
    truthData: (list) Ground truth data expected by the generator.
  Out:
    JSON and keras files of the saved model.
  """
  if args.validation != 0.0 and args.validation != 0.5:
    raise ValueError("Only 0.0 and 0.5 validation splits are supported for "
        "stateful models (b/c the batch_input_shape changes between training "
        "and validation data.")

  volumesPerBatch = int(args.batchSize)
  trainingBatchSize = volumesPerBatch - (volumesPerBatch * args.validation)
  if args.loadModel:
    modelPath = os.path.join(_DEFAULT_MODEL_DIR, args.loadModel+".json")
    with open(modelPath, "r") as infile:
      model = model_from_json(json.load(infile))
    model.compile(optimizer="adam", loss="mse")
    model.load_weights(os.path.join(_DEFAULT_MODEL_DIR, args.loadModel+".keras"))
    print "Model loaded from", modelPath
  else:
      if(args.usevgg == False):
          model = CNN_Self(volumesPerBatch=trainingBatchSize,
                       timesteps=args.timesteps,
                       cameraFormat=(_VIDEO_CHANNELS, _VIDEO_HEIGHT, _VIDEO_WIDTH),
                       verbosity=args.verbosity)
      else:
          model = VGG19(volumesPerBatch=trainingBatchSize,
                       timesteps=args.timesteps,
                       cameraFormat=(_VIDEO_CHANNELS, _VIDEO_HEIGHT, _VIDEO_WIDTH),
                       verbosity=args.verbosity)
         
  logging = [LossHistory()] if args.verbosity else []

  # Setup dir for model checkpointing
  if not os.path.exists(_DEFAULT_MODEL_DIR):
    os.makedirs(_DEFAULT_MODEL_DIR)
  checkpointConfig = os.path.join(_DEFAULT_MODEL_DIR, "steer_checkpoint.json")
  checkpointWeights = os.path.join(_DEFAULT_MODEL_DIR, "steer_checkpoint.keras")

        
  # Train the model on this batch
  print "Starting training..."
  for epoch in xrange(args.epochs):
    # Iterate through epochs explicitly b/c Keras fit_generator doesn't yield data as expected
      if args.verbosity > 0:
          print "\nTraining epoch {} of {}".format(epoch, args.epochs-1)
      End = 0
      while(End!=TrainCnt):
          Xtrain,Ytrain,End = getDataBatch(TrainCnt,End, args.databatchsize)
          print("End")
      
          history = model.fit(  # TODO: decay learning rate
                    Xtrain,
                    Ytrain,
                    batch_size=volumesPerBatch,
                    shuffle=True,
                    nb_epoch=1,
                    callbacks=logging,
                    validation_split=args.validation,
                    verbose=1)
          if args.verbosity:
            # Not using Keras's training loss print out b/c it always shows epoch 0
            print "Training loss =", history.history["loss"][0]


          print("current epoch ="+ str(epoch))
          EVal = TrainCnt
      while(EVal!=TrainCnt+150):
          XVal,YVal,EVal = getDataBatch(TrainCnt+150,EVal, args.databatchsize)
          predictedSteeringAngle = []  # TODO: preallocate arrays for len(truthData)
          truthSteeringAngle = []
          predictedSteeringAngle.extend(model.predict(XVal,batch_size=volumesPerBatch,verbose=1).flatten().astype(float))
          truthSteeringAngle.extend(YVal.flatten())
          predictedSteeringAngle = np.asarray(predictedSteeringAngle)
          truthSteeringAngle = np.asarray(truthSteeringAngle)
          rmse = ( np.linalg.norm(predictedSteeringAngle - truthSteeringAngle) /
                     np.sqrt(len(truthSteeringAngle)) )
          print "Finished validation, with a RMSE =", rmse
  
         #Checkpoint the model in case training breaks early
      if args.verbosity > 0:
          print "\nEpoch {} complete, checkpointing the model.".format(epoch)
          model.save_weights(checkpointWeights, True)
          with open(checkpointConfig, "w") as outfile:
              json.dump(model.to_json(), outfile)

  print "\nTraining complete, saving model weights and configuration files."
  model.save_weights(os.path.join(_DEFAULT_MODEL_DIR, "steer.keras"), True)
  with open(os.path.join(_DEFAULT_MODEL_DIR, "steer.json"), "w") as outfile:
    json.dump(model.to_json(), outfile)
  print "Model saved as .keras and .json files in", _DEFAULT_MODEL_DIR



def _runTest(args, images, truthData, TestCnt,TrainCnt):
  """
  Load a serialized model, run inference on a video stream, write out and
  compare the predictions to the given data labels.

  Args:
    (See the script's command line arguments.)
    frameGen: (generator) Yields X and Y data batches to train on.
    truthData: (list) Ground truth data expected by the generator.
  Out:
    JSON of prediction results, as a list of [time, speed] items.
    Displays line plot of predicted and truth speeds.
  """
  modelName = args.loadModel if args.loadModel else "steer"
  try:
    modelPath = os.path.join(_DEFAULT_MODEL_DIR, "steer_checkpoint"+".json")
    with open(modelPath, "r") as infile:
      model = model_from_json(json.load(infile))
  except:
    print "Could not load a saved model JSON from", modelPath
    raise

  # Serialized to JSON only preserves model archtecture, so we need to recompile
  model.compile(optimizer="adam", loss="mse")
  model.load_weights(os.path.join(_DEFAULT_MODEL_DIR, "steer_checkpoint"+".keras"))

  # Run inference
  


    
  predictedSteeringAngle = []  # TODO: preallocate arrays for len(truthData)
  truthSteeringAngle = []
  End = TrainCnt
  while(End!=TrainCnt+TestCnt):
      
      Xtest,Ytest,End = getDataBatch(TrainCnt+TestCnt,End, args.databatchsize)
      print("End")
      predictedSteeringAngle.extend(model.predict(Xtest).flatten().astype(float))
      truthSteeringAngle.extend(Ytest.flatten())

  resultsPath = os.path.join(_DEFAULT_MODEL_DIR, "steer_test.json")
  with open(resultsPath, "w") as outfile:
    json.dump(list(predictedSteeringAngle), outfile)
  print "Test results written to", resultsPath

  # Calculate the root mean squared error. We expect the data labels to cover
  # the full video that the model just ran prediction on.
  predictedSteeringAngle = np.asarray(predictedSteeringAngle)
  truthSteeringAngle = np.asarray(truthSteeringAngle)
  print(predictedSteeringAngle)
  print("##################################################")
  print(truthSteeringAngle)
  rmse = ( np.linalg.norm(predictedSteeringAngle - truthSteeringAngle) /
           np.sqrt(len(truthSteeringAngle)) )
  print "Finished testing, with a RMSE =", rmse

  if args.verbosity > 0:
    # Show line plot of results
    # TODO: use plotly
    plt.plot(truthSteeringAngle,'r')
    plt.plot(predictedSteeringAngle,'b')
    plt.savefig('Testing.png', dpi=100)



if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-v", "--ImagesPath",
                      type=str,
                      default="/media/kaustubh/Linux_D/Challenge2/dataset/",
                      help="Path to center cam image.")
  parser.add_argument("-u", "--ImageSeqFile",
                      type=str,
                      default="/media/kaustubh/Linux_D/Challenge2/camera.csv",
                      help="Path to image analysis file.")
  parser.add_argument("-d", "--dataPath",
                      type=str,
                      default="/media/kaustubh/Linux_D/Challenge2/steering.csv",
                      help="Path to JSON of ground truth speeds.")
  parser.add_argument("--batchSize",
                      type=int,
                      default=30,
                      help="Frames per batch yielded by the data generator.")
  parser.add_argument("-e", "--epochs",
                      type=int,
                      default=1000,
                      help="Number of epochs.")
  parser.add_argument("-t", "--timesteps",
                      type=int,
                      default=5,
                      help="Number of consecutive video frames per CNN volume.")
  parser.add_argument("-db", "--databatchsize",
                      type=int,
                      default=90,
                      help="Number of images preprocessed at a time. Not to be confused with batchsize")
  parser.add_argument("--test",
                      default=True,
                      action="store_true",
                      help="Run test phase (using saved model).")
  parser.add_argument("--skipTraining",
                      default=False,
                      action="store_true",
                      help="Bypass training phase.")
  parser.add_argument("--validation",
                      type=float,
                      default=0,
                      help="Portion of training data for validation split.")
  parser.add_argument("--loadModel",
                      type=str,
                      default="",
                      help="Load a specific model to train or test.")
  parser.add_argument("--usevgg",
                      default=False,
                      action="store_true",
                      help="Use VGG19 model")
  parser.add_argument("--verbosity",
                      type=int, 
                      default=1,
                      help="Level of printing stuff to the console.")
  args = parser.parse_args()

  # Train or test; experiment setup is done for both b/c we need to reset data
  # objects if running test immediately after training (on the same data).
  images, truthData = DataPreparation(args.ImagesPath,args.ImageSeqFile, args.dataPath)
  
  volumesPerBatch = int(args.batchSize)
  
  Total = len(images)
  #Total = 1000
  TestCnt = 600
  TrainCnt = 45000
  
  if not args.skipTraining:
    _runTrain(args,images, truthData,TrainCnt,TestCnt)
  if args.test:
    _runTest(args, images, truthData, TestCnt,TrainCnt)
