# -*- coding: utf-8 -*-

import numpy as np
from PIL import Image
import glob
import matplotlib.pyplot as plt 
import classification

def import_detect_images():
    image_list = []

    for filename in glob.glob('dataset/detection-images/*.jpg'): 
        im=Image.open(filename)
        image_list.append(im)
 
    return image_list

def whiteness_filter(image, width, height, ratio): #filter out images that are mostly white (don't contain characters)
    threshold = ratio*255*width*height
    im_array = np.array(image)
    if (im_array.sum() > threshold):
        return False #discard white image
    else:
        return True #keep image
    
def threshold_filter(images, coordinates, classes, scores, threshold):
    mask = [np.array(scores) > 0.6][0]

    f_scores = np.array(scores)[mask].tolist()
    f_classes = np.array(classes)[mask].tolist()
    f_coordinates = np.array(coordinates)[mask].tolist()
    f_images = []
    for i in range(0, len(images)):
        if (mask[i]):
            f_images.append(images[i])
    
    return f_images, f_coordinates, f_classes, f_scores

def box_area(pmin,pmax):
    return (pmax[0] - pmin[0])*(pmax[1] - pmin[1])


def iou(coordinate1, coordinate2):
    
    #convert to x and y coordinates (x1, y1, x2, y2)
    box1 = [coordinate1[0], coordinate1[1], coordinate1[0] + coordinate1[2], coordinate1[1] + coordinate1[2]]
    box2 = [coordinate2[0], coordinate2[1], coordinate2[0] + coordinate2[2], coordinate2[1] + coordinate2[2]]
    
    min_point = np.zeros((1,2))[0]
    max_point = np.zeros((1,2))[0]
    
    #min_point = max(xmin,ymin)
    min_point[0] = max(box1[0], box2[0])
    min_point[1] = max(box1[1], box2[1])
    
    #max_point = min(xmax,ymax)
    max_point[0] = min(box1[2], box2[2])
    max_point[1] = min(box1[3], box2[3])
    
    U = 1.
    I = 0
    if (max_point[0] - min_point[0] >= 0) and (max_point[1] - min_point[1] >= 0):
        I = box_area(min_point, max_point)
        U = box_area(box1[0:2], box1[2:4]) 
    ans = I/U
    
    return ans

def non_max_supression(samples, coordinates, classes, scores, iou_threshold=0.3, max_boxes=10): #prevent overlapping detections
    nms_indices = []
    # Use iou() to get the list of indices corresponding to boxes you keep    
    scores_indexes = np.array(scores).argsort().tolist()
    
    while len(scores_indexes):
        idx_max = scores_indexes.pop()
        nms_indices.append(idx_max)
        tmp = scores_indexes[:]
        for idx in scores_indexes:
            iouScore = iou(coordinates[idx_max], coordinates[idx])
            if  iouScore >= iou_threshold:
                tmp.remove(idx)
        scores_indexes = tmp[:]
    # Use index arrays to select only nms_indices from scores, boxes and classes
    
    if len(nms_indices) > max_boxes:
        nms_indices = nms_indices[0:max_boxes]
    
    print(nms_indices)
    f_scores = np.array(scores)[nms_indices].tolist()
    f_classes = np.array(classes)[nms_indices].tolist()
    f_coordinates = np.array(coordinates)[nms_indices].tolist()
    f_images = []
    for i in range(0, len(nms_indices)):
        f_images.append(samples[nms_indices[i]])
            
    return f_images, f_coordinates, f_classes, f_scores



def plot_boxes (image, coordinates, classes, scores):
    plt.figure()
    plt.imshow(image)
    
    class_names = []
        
    for letter in range(97,123):
        class_names.append(chr(letter))

    legend_map = {}
    for i in reversed(list(range(len(classes)))):
        c = classes[i]
        predicted_class = class_names[c]
        box = [coordinates[i][0], coordinates[i][1], coordinates[i][0] + coordinates[i][2], coordinates[i][1] + coordinates[i][2]]
        score = scores[i]
        label = '{} {:.2f}'.format(predicted_class, score)
        left, top, right, bottom = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
        plt.text(left, top, label, fontsize=12)
        print(label, (left, top), (right, bottom))
        x = [left, left, right, right, left]
        y = [top, bottom, bottom, top, top]
        line, = plt.plot(x,y)
        legend_map[predicted_class] = line
        
        
    classes_list = list(legend_map.keys())
    values = [legend_map[k] for k in classes_list]
    plt.legend(values, classes_list)

def detect_single_image(im, model, stride, window_sizes):
    im_w, im_h = im.size
    samples = []
    coordinates = [] #(left, lower, window size)
    for w_size in window_sizes:
        for i in range(0,(im_h - w_size + stride),stride):
            for j in range(0,(im_w - w_size + stride),stride):
                box = (j, i, j + w_size, i + w_size) #(left, upper, right, lower)
                im_crop = im.crop(box)
                #filter of white area
                if (whiteness_filter(im_crop, w_size, w_size, 0.7)):
                    samples.append(im_crop)
                    coordinates.append([j,i,w_size])      
    
    scores = []
    for i in range(0,len(samples)):
        samples[i] = samples[i].resize((20,20))
        
    samples = classification.transform_data(samples)
    classes, scores_vec = classification.svm_predict(model, samples)
    
    classes = [int(i) for i in classes]
    for i in range(0,len(scores_vec)):
        scores.append(scores_vec[i][int(classes[i])])
    #threshold
    samples, coordinates, classes, scores = threshold_filter(samples, coordinates, classes, scores, 0.6)
    
    #non-max supression
    samples, coordinates, classes, scores = non_max_supression(samples, coordinates, classes, scores, max_boxes = 50)
      
    #plotting boxes
    plot_boxes (im, coordinates, classes, scores)   


def main():
    #define hyperparameters stride, window_sizes[]
    stride = 2
    window_sizes = [20]
    
    #training the svm
    X, Y = classification.get_data(True)
    model = classification.svm_train(X, Y)
    
    #import the images
    images = import_detect_images()
    
    #sliding window
    for im in images:
        detect_single_image(im, model, stride, window_sizes)

if __name__ == "__main__":
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    