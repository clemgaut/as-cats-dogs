# -*- coding: utf-8 -*-
"""
First file to perform training in dog vs cats kaggle challenge.
"""
import random

import time

import numpy as np

from skimage import io
from skimage import color

from sklearn import cross_validation
from sklearn import svm

import glob


def list_train_images(n=1000):
    """
    List n training images chosen at random.
    """
    image_filenames = glob.glob("./train/*.jpg")
    image_filenames.sort()
    random.seed(42)
    random.shuffle(image_filenames)
    return image_filenames[0:n-1]
    
def build_patches(channel, N):
    """
    Build patches for the given channel.
    """
    patches = []
    
    #It's a square image so same size for width and heigth
    patch_size = int(channel.shape[0]/N)
    
    for i in range(N):
        for j in range(N):
            patch = channel[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
            patches.append(patch)
            
    return patches
    
def hsv_to_feature(hsv,N,C_h,C_s,C_v):
  """ Takes an hsv picture and returns a feature vector for it.
  The vector is built as described in the paper 'Machine Learning Attacks Against the Asirra CAPTCHA' """  
  res = np.zeros((N,N,C_h,C_s,C_v))
  cell_size= 250/N
  h_range = np.arange(0.0,1.0,1.0/C_h)
  h_range = np.append(h_range,1.0)
  s_range = np.arange(0.0,1.0,1.0/C_s)
  s_range = np.append(s_range,1.0)
  v_range = np.arange(0.0,1.0,1.0/C_v)
  v_range = np.append(v_range,1.0)
  for i in range(N):
    for j in range(N):
      cell= hsv[i*cell_size:i*cell_size+cell_size,j*cell_size:j*cell_size+cell_size,:]
      # check for h
      for h in range(C_h):
        h_cell = np.logical_and(cell[:,:,0]>=h_range[h],cell[:,:,0]<h_range[h+1])
        for s in range(C_s): 
          s_cell = np.logical_and(cell[:,:,1]>=s_range[s],cell[:,:,1]<s_range[s+1])
          for v in range(C_v):
            v_cell = np.logical_and(cell[:,:,2]>=v_range[v],cell[:,:,2]<v_range[v+1])
            gesamt = np.logical_and(np.logical_and(h_cell,s_cell),v_cell)
            res[i,j,h,s,v] = gesamt.any()
  return np.asarray(res).reshape(-1)
    
    
def compute_color_feature_matrix(file_list, N, ch, cs, cv): 
    m = []
    for f in file_list:
        img = io.imread(f, as_grey=False)
        m.append(hsv_to_feature(color.rgb2hsv(img), N, ch, cs, cv))
        
    return m
        
def load_matrices(file_list, only_F1=True):
    try:
        F1 = np.load("F1.npy")
    except IOError:
        F1 = compute_color_feature_matrix(file_list,1,10,10,10)
        np.save("F1",F1)
        
    if only_F1:
        return F1
    try:
        F2 = np.load("F2.npy")
    except IOError:
        F2 = compute_color_feature_matrix(file_list,3,10,8,8)
        np.save("F2",F2)
    try:
        F3 = np.load("F3.npy")
    except IOError:
        F3 = compute_color_feature_matrix(file_list,5,10,6,6)
        np.save("F3",F3)
        
    return F1,F2,F3
    
def compute_labels(file_list):
    lbl = np.zeros(len(file_list),dtype=np.int32)
    
    for (i,f) in enumerate(file_list):
        if "dog" in f:
            lbl[i] = -1
        else:
            lbl[i] = 1
            
    return lbl
    
def classify_color_feature(F,y):
    start = time.time()
    clf = svm.SVC(kernel='rbf',gamma=0.001)
    scores = cross_validation.cross_val_score(clf, F, y, cv=5) 
    time_diff = time.time() - start 
    print "Accuracy: %.1f  +- %.1f   (calculated in %.1f seconds)"   % (np.mean(scores)*100,np.std(scores)*100,time_diff)
    
if __name__ == "__main__":
    file_list = list_train_images(5000) 
    
    F1 = load_matrices(file_list)
    
    y = compute_labels(file_list)
    
    classify_color_feature(F1,y)
    
    
