import datetime
import os, sys, shutil
import numpy as np
from numpy import loadtxt
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import optimizers, applications
from tensorflow.keras.applications import VGG19
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras import models
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras import optimizers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import TensorBoard
import time
import tensorflow as tf
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform




model=tf.keras.models.load_model("Xception_model/mod.h5")#,custom_objects={'PruneLowMagnitude': prune_low_magnitude_output()})
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]#POST-Trainig quantization (default:converter wimm try to figure out the best balance between size and latency)
                                        #.OPTIMIZE_FOR_SIZE (optimize for size)
                                        #.OPTIMIZE_FOR_LATENCY (optimize for latency) Latency is a networking term to describe the total time it takes a
                                                                                    #data packet to travel from one node to another(temps de transmission)
#converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
tflite_model=converter.convert()
tflite_model_file='converted_modell_Xc.tflite'
with open(tflite_model_file,"wb") as f:
    f.write(tflite_model)
