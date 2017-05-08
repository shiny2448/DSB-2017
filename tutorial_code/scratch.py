# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 10:19:04 2017

@author: SF995864
"""

import numpy as np
import matplotlib.pyplot as plt
import LUNA_train_unet

working_path = "../../DSB3Tutorial/output_with_orig_img/"

pred = np.load(working_path+'pred_masks.npy')
true = np.load(working_path+"testMasks.npy")
img = np.load(working_path+"testImages.npy")
orig = np.load(working_path+"testImages_uncropped.npy")

for i in range(len(pred)):
    plt.figure(figsize=(24, 6))
    
    plt.subplot(141)
    plt.title('Input Slice')
    plt.axis('off')
    plt.imshow(orig[i,0], cmap=plt.cm.bone)
    
    plt.subplot(142)
    plt.title('Lung Region')
    plt.axis('off')
    plt.imshow(img[i,0], cmap=plt.cm.bone)
    
    plt.subplot(143)
    plt.title('True Mask')
    plt.axis('off')
    plt.imshow(true[i,0], cmap=plt.cm.bone)
    plt.clim(0,1/255)
    
    plt.subplot(144)
    plt.title('Predicted Mask')
    plt.axis('off')
    plt.imshow(pred[i,0], cmap=plt.cm.bone)
    plt.clim(0,1/255)
    print(LUNA_train_unet.dice_coef_np(true[i,0], pred[i,0]))
    
    plt.savefig('output/' + str(i) + '.png')
    plt.close()
