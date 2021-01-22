# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 17:12:27 2020

@author: Hp
"""

import cv2
import imutils
import numpy as np
from PIL import Image

from collections import Counter

import time
import numpy as np
import os
import matplotlib.pyplot as plt

os.chdir(r'G:\ASL\asl-alphabet\train\A')

img = cv2.imread('A689.jpg')
plt.imshow(img)
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
plt.imshow(gray,cmap=plt.get_cmap("gray"))
ret,thresh1 = cv2.threshold(gray,125,255,cv2.THRESH_BINARY)
plt.imshow(thresh1,cmap=plt.get_cmap("gray"))


cv2.imwrite('test01.jpg',gray)
