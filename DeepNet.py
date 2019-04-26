# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 16:17:03 2019

@author: silvi
"""
import numpy as np
import util


images, labels = util.import_dataset()
X_train = np.array(util.flatten_image_list (images))
Y_train = np.array(util.onehot_encode(labels))

X_train, Y_train, X_test, Y_test = util.train_val_split(X_train, Y_train, 0.2)

X_train, Y_train, X_val, Y_val = util.train_val_split(X_train, Y_train, 0.1)


