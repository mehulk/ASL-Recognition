# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 17:04:57 2020

@author: Mehul
"""

import tensorflow as tf
print(tf.__version__)

from PIL import Image

from keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import os
import h5py
import cv2
import imutils
import numpy as np

from collections import Counter

import time
import numpy as np
import os
import matplotlib.pyplot as plt

os.chdir(r'G:\ASL\ASL-Recognition\ASL-Recognition')

#%%

IMG_SIZE = 200

model = load_model('asl_inception_4_orig.h5')
print("Model Loaded")

model.summary()

#%%

out_label = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']
pre = []

s=''
cchar=[0,0]
c1=''

# initialize weight for running average
aWeight = 0.5

# get the reference to the webcam
camera = cv2.VideoCapture(0)

data_generator = ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True)

# region of interest (ROI) coordinates

IMAGE_SIZE = 200
CROP_SIZE = 400
top, right, bottom, left = 100, 100, 300, 300

# initialize num of frames
num_frames = 0

flag=0
flag1=0

i = 0
mem = ''
res = ''
consecutive = 0
sequence = ''
prediction_probability = 0

#%%

while(True):
    # get the current frame
    (grabbed, frame) = camera.read()
 
    # resize the frame
    frame = imutils.resize(frame, width=800)
    img = Image.fromarray(frame, 'RGB')    

    # flip the frame 
    frame = cv2.flip(frame, 1)

    # clone the frame
    clone = frame.copy()

    # get the height and width of the frame
    (height, width) = frame.shape[:2]

    # get the ROI
    img = frame[top:bottom, right:left]

    # blur it
    img = cv2.GaussianBlur(img, (7, 7), 0)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
    #img=np.array(img)/255.
    #plt.imshow(img)
    #test_data = img

    orig = img
    data = img.reshape(1,IMG_SIZE,IMG_SIZE,3)
    frame_for_model = data_generator.standardize(np.float64(data))
    
    if i==4:
         model_out = np.array(model.predict([frame_for_model]))
         prediction_probability = model_out[0, model_out.argmax()]
         res = out_label[np.argmax(model_out)]
         i = 0
         if prediction_probability > 0.2:
             if mem == res:
                 consecutive += 1
             else:
                 consecutive = 0  
             if consecutive == 3 and res not in ['nothing']:
                if res == 'space':
                    sequence += ' '
                elif res == 'del':
                    sequence = sequence[:-1]
                else:
                    sequence += res
                consecutive = 0
    i += 1

    # draw the segmented hand
    cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)

    if prediction_probability > 0.5:
        cv2.putText(clone, 'High %s' % (res.upper()), (300,150), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)
        mem = res
        
    elif prediction_probability > 0.2 and prediction_probability <= 0.5:
        cv2.putText(clone, 'Maybe %s' % (res.upper()), (300,150), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)
        mem = res 
    
    cv2.putText(clone, '%s' % (sequence.upper()), (300,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText(clone, '%s ' % (str(s)), (10, 60), cv2.FONT_HERSHEY_PLAIN,2,(255, 255, 255))
    
    # increment the number of frames
    num_frames += 1
    
    # display the frame with segmented hand
    cv2.imshow("Video Feed", clone)

    # observe the keypress by the user
    keypress = cv2.waitKey(1) & 0xFF

    # if the user pressed "q", then stop looping
    if keypress == ord("q"):
        break

    elif keypress == 27:
        break

# free up memory
camera.release()
cv2.destroyAllWindows()

