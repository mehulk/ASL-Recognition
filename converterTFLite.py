# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 15:43:04 2020

@author: Hp
"""

import tensorflow as tf
from keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import os

os.chdir(r'G:\ASL')
#%%
import cv2 
img = cv2.imread('A1.jpg')
 #%%
 print(img.shape)



#%%

model = load_model('asl_inception_4_orig.h5')
print("Model Loaded")
#%%


converter = tf.lite.TFLiteConverter.from_keras_model_file('asl_inception_4_orig.h5')
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)

#%%
interpreter = tf.lite.Interpreter(model_path="converted_model.tflite")
interpreter.allocate_tensors()

# Print input shape and type
print(interpreter.get_input_details()[0]['shape'])  # Example: [1 224 224 3]
print(interpreter.get_input_details()[0]['dtype'])  # Example: <class 'numpy.float32'>

# Print output shape and type
print(interpreter.get_output_details()[0]['shape'])  # Example: [1 1000]
print(interpreter.get_output_details()[0]['dtype'])