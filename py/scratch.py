# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 18:01:41 2017

@author: bennettng
"""

import numpy as np
import pandas as pd
import dicom
import os

from scipy import ndimage
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
mask_lungs = segmentation.segment_lung_mask(img_resamp, True)
mask_lungs = ndimage.binary_dilation(mask_lungs) # dilate mask   
img_lungs = img_resamp * mask_lungs

imutil.plot_collage(mask_lungs)