# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 18:01:41 2017
https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial
https://www.kaggle.com/arnavkj95/data-science-bowl-2017/candidate-generation-and-luna16-preprocessing/

@author: bennettng
"""

import numpy as np
import pandas as pd
import dicom
import os
import matplotlib.pyplot as plt

from scipy import ndimage
from skimage import morphology, measure
import imutil
import segmentation

# load slices from a given path
def load_scan(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices


INPUT_FOLDER = '../data/sample_images/'
patients = os.listdir(INPUT_FOLDER)
patients.sort()
pt = patients[1]
print('processing patient: ', pt)

slices = load_scan(INPUT_FOLDER + pt)
img = imutil.get_pixels_hu(slices)
img_resamp, spacing = imutil.resample(img, slices, [1,1,1])
mask_lungs_notfilled = segmentation.segment_lung_mask(img_resamp, False)
mask_lungs_filled = segmentation.segment_lung_mask(img_resamp, True)
mask_lungs_diff = mask_lungs_filled - mask_lungs_notfilled
imutil.plot_3d(mask_lungs_diff, 0)

# remove two largest connected components (blood vessels)
label_scan = morphology.label(mask_lungs_diff)
rprops = measure.regionprops(label_scan)
areas = [r.area for r in rprops]
areas.sort()

for r in rprops:
    max_x, max_y, max_z = 0, 0, 0
    min_x, min_y, min_z = 1000, 1000, 1000
    
    for c in r.coords:
        max_z = max(c[0], max_z)
        max_y = max(c[1], max_y)
        max_x = max(c[2], max_x)
        
        min_z = min(c[0], min_z)
        min_y = min(c[1], min_y)
        min_x = min(c[2], min_x)
    if (min_z == max_z or min_y == max_y or min_x == max_x or r.area > areas[-3]):
        for c in r.coords:
            mask_lungs_diff[c[0], c[1], c[2]] = 0
    else:
        index = (max((max_x - min_x), (max_y - min_y), (max_z - min_z))) / (min((max_x - min_x), (max_y - min_y) , (max_z - min_z)))

imutil.plot_3d(mask_lungs_diff, 0)

#mask_lungs_diff = ndimage.binary_dilation(mask_lungs_diff) # dilate mask   
#img_lungs = img_resamp * mask_lungs_filled
#mask_nodules = (img_lungs > -400) & img_lungs
#imutil.plot_collage(mask_nodules)