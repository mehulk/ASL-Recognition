""" 17/04/2020"""

import tensorflow as tf
print(tf.__version__)
#%%
from PIL import Image


#%%
import tflearn
from tflearn.layers.estimator import regression
from keras import Model
from tensorflow.keras.models import load_model
import os
os.chdir(r'G:\ASL')

#%%

IMG_SIZE = 200
#LR = 1e-3

model = load_model('asl_inception_4_orig.h5')
print("Model Loaded")

model.summary()

#%%

import cv2
import imutils
import numpy as np

from collections import Counter

import time
import numpy as np
import os
import matplotlib.pyplot as plt

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

# region of interest (ROI) coordinates
top, right, bottom, left = 100, 100, 300, 300

# initialize num of frames
num_frames = 0

flag=0
flag1=0
#%%

while(True):
    # get the current frame
    (grabbed, frame) = camera.read()

    # resize the frame
   # frame = imutils.resize(frame, width=700)
   # img = Image.fromarray(frame, 'RGB')    

    # flip the frame so that it is not the mirror view
    frame = cv2.flip(frame, 1)

    # clone the frame
    clone = frame.copy()

    # get the height and width of the frame
    (height, width) = frame.shape[:2]

    # get the ROI
    img = frame[top:bottom, right:left]

    # convert the roi to grayscale and blur it
    #gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    #gray = cv2.GaussianBlur(gray, (7, 7), 0)

    
    # cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
    
    # img=gray
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img=cv2.imread("240fn.jpg",cv2.IMREAD_GRAYSCALE)
    # img=cv2.cvtColor(bw_image,cv2.COLOR_BGR2GRAY)
    
    img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
    img=np.array(img)/255.
    plt.imshow(img)
    test_data =img

    orig = img
    data = img.reshape(1,IMG_SIZE,IMG_SIZE,3)
    #model_out = model.predict([data])[0]
    model_out = model.predict([data])[0]
    
    # print(model_out)
   # model_out = model.predict([data])[0]
        # print(model_out)
    pnb=np.argmax(model_out)
    print(str(np.argmax(model_out))+" "+str(out_label[pnb]))

    pre.append(out_label[pnb]) 


    cv2.putText(clone,
           '%s ' % (str(out_label[pnb])),
           (450, 150), cv2.FONT_HERSHEY_PLAIN,5,(0, 255, 0))

            

            
        


    # draw the segmented hand
    cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)

    cv2.putText(clone,
                   '%s ' % (str(s)),
                   (10, 60), cv2.FONT_HERSHEY_PLAIN,3,(255, 255, 255))

    # increment the number of frames
    num_frames += 1
    # time.sleep(.3)
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
