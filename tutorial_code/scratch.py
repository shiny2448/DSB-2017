# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 10:19:04 2017

@author: SF995864
"""

import numpy as np
import matplotlib.pyplot as plt
import LUNA_train_unet

working_path = "../../DSB3Tutorial/output/"

pred = np.load('pred_masks.npy')
true = np.load(working_path+"testMasks.npy")
img = np.load(working_path+"testImages.npy")

for i in range(15):
    plt.figure()
    plt.subplot(131)
    plt.title('True')
    plt.axis('off')
    plt.imshow(true[i,0], cmap=plt.cm.bone)
    plt.clim(0,1/255)
    
    plt.subplot(132)
    plt.title('Predicted')
    plt.axis('off')
    plt.imshow(pred[i,0], cmap=plt.cm.bone)
    plt.clim(0,1/255)
    print(LUNA_train_unet.dice_coef_np(true[i,0], pred[i,0]))
    
    plt.subplot(133)
    plt.title('Image')
    plt.axis('off')
    plt.imshow(img[i,0], cmap=plt.cm.bone)