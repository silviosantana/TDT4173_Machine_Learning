# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 11:40:21 2019

@author: silvi
"""

import numpy as np
from PIL import Image
import glob

def onehot_encode(Y, n_classes=26):
    onehot = np.zeros((len(Y), n_classes))
    onehot[np.arange(0, len(Y)), Y] = 1
    return onehot

def import_dataset():
    image_list = []
    label_list = []
    alphabet = []
    
    for letter in range(97,123):
        alphabet.append(chr(letter))
    
    label = 0
    for letter in alphabet:
        for filename in glob.glob('dataset/chars74k-lite/' + letter + '/*.jpg'): #assuming gif
            im=Image.open(filename)
            image_list.append(im)
            label_list.append(label)
        
        label += 1
    
    return image_list, label_list


def flatten_image_list (image_list):
    flatten_list = []
    for i in range(0,len(image_list)):
        flatten_list.append(np.array(image_list[i]).flatten().tolist())
        
    return flatten_list

X_train, Y_train = import_dataset()
X_flatten = flatten_image_list (X_train)
Y_hot = onehot_encode(Y_train)
    


    
    
