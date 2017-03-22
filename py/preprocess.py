# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import dicom
import os

import imutil
import segmentation

# constants 
INPUT_FOLDER = '../data/sample_images/'

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

def processPatient(name):
    print('processing patient: ', name)
    slices = load_scan(INPUT_FOLDER + name)
    pixels = imutil.get_pixels_hu(slices)
    px_resamp, spacing = imutil.resample(pixels, slices, [1,1,1])
    segmented_lungs_filled = segmentation.segment_lung_mask(px_resamp, True)
    return segmented_lungs_filled
    print('done')

def main():    
    patients = os.listdir(INPUT_FOLDER)
    patients.sort()
    for pt in patients:
        processPatient(pt)

if __name__ == '__main__':
    main()