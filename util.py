# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 11:40:21 2019

@author: silvi
"""

import numpy as np
from PIL import Image
import glob
import cv2

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
            im = cv2.imread(filename,0)
            image_list.append(im)
            label_list.append(label)
        
        label += 1
    
    return image_list, label_list


def flatten_image_list (image_list):
    flatten_list = []
    for i in range(0,len(image_list)):
        flatten_list.append(np.array(image_list[i]).flatten().tolist())
        
    return flatten_list


def shuffle_data(X, Y):
    idx = np.arange(0, X.shape[0]);
    np.random.shuffle(idx);

    X, Y = X[idx], Y[idx];
    
    return X, Y;

def train_val_split(X, Y, val_percentage):
  """
    Selects samples from the dataset randomly to be in the validation set. Also, shuffles the train set.
    --
    X: [N, num_features] numpy vector,
    Y: [N, 1] numpy vector
    val_percentage: amount of data to put in validation set
  """
  Y = np.reshape(Y,(len(Y),1))
  X = np.append(X, Y, axis=1)
  np.random.shuffle(X) 
  X_train = []
  Y_train = []
  X_val = []
  Y_val = []
  for i in range(0, int(len(X)*val_percentage)):
      X_train.append(X[i][:-1])
      Y_train.append(X[i][-1:])

  for i in range(len(X_train), len(X)):    
      X_val.append(X[i][:-1])
      Y_val.append(X[i][-1:])
      
  Y_train = np.reshape(Y_train,(len(Y_train),))
  Y_val = np.reshape(Y_val,(len(Y_val),))
  return X_train, Y_train, X_val, Y_val




    
    
