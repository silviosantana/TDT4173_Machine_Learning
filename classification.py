# -*- coding: utf-8 -*-

import numpy as np
import util
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from skimage import feature as ft
from sklearn import svm
import skimage as sk
#from skimage import transform
#from skimage import util
from scipy import ndarray
import random
from sklearn.neural_network import MLPClassifier


def random_rotation(image_array: ndarray):
    # pick a random degree of rotation between 25% on the left and 25% on the right
    image_array = np.resize(image_array,(20,20))
    random_degree = random.uniform(-90, 90)
    return np.reshape(sk.transform.rotate(image_array, random_degree),(400,))

def random_noise(image_array: ndarray):
    # add random noise to the image
    image_array = np.resize(image_array,(20,20))
    return np.reshape(sk.util.random_noise(image_array),(400,))

def augmentation(X_train, Y_train):
    l = len(X_train)
    X_train = np.array(X_train)
    for i in range(0, l,20):
        tmp = np.reshape(np.array(random_rotation(X_train[i])),(1,400))
        X_train = np.append(X_train, tmp, axis = 0)
        Y_train = np.append(Y_train,Y_train[i])
        #X_train.append(ndarray.tolist(random_rotation(X_train[i])))
        #Y_train = np.append(Y_train,Y_train[i])
        tmp = np.reshape(np.array(random_noise(X_train[i])),(1,400))
        X_train = np.append(X_train, tmp, axis = 0)
        Y_train = np.append(Y_train,Y_train[i])
    return X_train, Y_train

def play(imageBytes):
    hogg = []
    for i in range(0, len(imageBytes)):
        arr = imageBytes[i]
        arr = np.resize(arr,(20,20))
        fd, arr = ft.hog(arr, orientations=8, pixels_per_cell=(10, 10),
                        cells_per_block=(1, 1), visualize=True, block_norm = 'L2-Hys')
        arr = np.reshape(arr,(400,))
        hogg.append(arr)
        
        
#    lbp = []
#    for i in range(0, len(imageBytes)):
#        arr = imageBytes[i]
#        arr = np.resize(arr,(20,20))
#        arr = ft.local_binary_pattern(arr, 8, 1, 'uniform')
#        n_bins = int(arr.max() + 1)
#        hist, _ = np.histogram(arr, density=True, bins=n_bins, range=(0, n_bins))
#        hist, _ = np.histogram(arr.ravel(),	bins=np.arange(0, 8 + 3),range=(0, 8 + 2))
#		# normalize the histogram
#        hist = hist.astype("float")
#        hist /= (hist.sum() + 1e-7)
#        
#        arr = np.reshape(arr,(400,))
#        lbp.append(hist)
    imageBytes = np.append(imageBytes, hogg, axis=1)
    #imageBytes = np.append(imageBytes, lbp, axis=1)
    imageBytes = StandardScaler().fit_transform(imageBytes)
    #pca = PCA(n_components=50)
    #return pca.fit_transform(imageBytes)
    return imageBytes


def transform_data(samples):
    n_samples = []
    for i in range(0, len(samples)):
        n_samples.append(np.array(samples[i]))
    imageBytes = np.array(util.flatten_image_list (n_samples))
    primarycomp = play(imageBytes)
    
    return primarycomp

def get_data(detection): 
    images, labels = util.import_dataset()
    imageBytes = np.array(util.flatten_image_list (images))
    
    if detection:
        #imageBytes, labels = augmentation(imageBytes, labels)
        primarycomp = play(imageBytes)
        return primarycomp, labels
    else:
        primarycomp = play(imageBytes)
        X_train, Y_train, X_test, Y_test = util.train_val_split(primarycomp, labels, 0.8)
        return X_train, Y_train, X_test, Y_test

def svm_train(X_train, Y_train):
    svmC = svm.SVC(gamma='scale', probability=True)
    svmC.fit(X_train, Y_train)
    
    return svmC

def svm_predict(svmC, X_test):
    svm_y_pred = svmC.predict(X_test)
    confidence = svmC.predict_proba(X_test)  
    return svm_y_pred, confidence
    
def mlp_train(X_train, Y_train):
    mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(100, 50), random_state=1)
    mlp.fit(X_train, Y_train)
    return mlp

def mlp_predict(mlp, X_test):
    mlp_y_pred = mlp.predict(X_test)
    confidence = mlp.predict_proba(X_test)  
    return mlp_y_pred, confidence

def rf_train(X_train, Y_train):
    rf = RandomForestClassifier(n_estimators=300,  random_state=0)
    rf.fit(X_train, Y_train)
    return rf

def rf_predict(rf, X_test):
    rf_y_pred = rf.predict(X_test)
    confidence = rf.predict_proba(X_test)  
    return rf_y_pred, confidence

def print_single_results(model, X_test, Y_test, alg):
    X_test_5 = X_test[0:5]
    Y_test_5 = Y_test[0:5]
    
    if (alg == 'svm'):
        pred, confidence = svm_predict(model, X_test_5)
    elif(alg == 'mlp'):
        pred, confidence = mlp_predict(model, X_test_5)
    else:
        pred, confidence = rf_predict(model, X_test_5)
    
    scores = []
    pred = [int(i) for i in pred]
    for i in range(0,len(confidence)):
        scores.append(confidence[i][int(pred[i])])
        
    print("Individual Predictions for " + alg)
    print("True Classes", Y_test_5)
    print("Predicted Classes: ", pred)
    print("Pred. confidence: ", scores)
    

def main():
    X_train, Y_train, X_test, Y_test = get_data(False)
    
    #X_train, Y_train = augmentation(X_train, Y_train)
    #X_train = play(X_train)
    #X_test = play(X_test)
    
    print("Training SVM model...")
    model = svm_train(X_train, Y_train)
    svm_y_pred, confidence = svm_predict(model, X_test)
    print("Accuracy for SVM is ", accuracy_score(Y_test,svm_y_pred)*100)
    
    print("Training MLP model...")
    mlp = mlp_train(X_train, Y_train)
    mlp_y_pred, confidence = mlp_predict(mlp, X_test)
    print("Accuracy for MLP is ", accuracy_score(Y_test,mlp_y_pred)*100)
    
    print("Training Randon Forest model...")
    rf = rf_train(X_train, Y_train)
    rf_y_pred, confidence = rf_predict(rf, X_test)
    print("Accuracy for Rand. Forest is ", accuracy_score(Y_test,rf_y_pred)*100)
    
    print_single_results(model, X_test, Y_test, 'svm')
    print_single_results(mlp, X_test, Y_test, 'mlp')
    print_single_results(rf, X_test, Y_test, 'Randon Forest')
    
if __name__ == "__main__":
    main()
    
    
    


 



