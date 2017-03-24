# -*- coding: utf-8 -*-

import numpy as np # linear algebra
import dicom
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

import LUNA_segment_lung_ROI
import LUNA_train_unet

INPUT_FOLDER = '../data/sample_images/'
OUTPUT_FOLDER = '../data/segmented_nodules/'

# Load the scans in given folder path
def get_slices(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))        
    return slices

if __name__ == "__main__":
    # find scan data
    patients = os.listdir(INPUT_FOLDER)
    patients.sort()
    print('-'*40)
    print('Prediciting node masks')
    print('-'*40)
    
    # load UNET nodule segmentation model
    unet = LUNA_train_unet.get_unet()
    unet.load_weights('./unet_trained.hdf5')
    
    # segment
    for i in range(len(patients)):
        print('Processing patient', str(i+1), 'of', str(len(patients)))
        patient = patients[i]
        slices = get_slices(INPUT_FOLDER + patient)
        nodules = []
        for count, s in enumerate(tqdm(slices)):
            img = s.pixel_array
            # segment lungs
            mask_lungs = LUNA_segment_lung_ROI.get_lung_mask(img)
            if sum(mask_lungs.flatten()) is 0:
                img_lungs = -1
            else:
                img_lungs = LUNA_segment_lung_ROI.apply_mask_normalize(img, mask_lungs)
            
            if img_lungs is not -1:
                # segment nodules
                mask_nodule = unet.predict(img_lungs.reshape([1,1,512,512]), verbose=0)[0]
                
                # TODO: ignore if no nodules found
                
                nodules.append(img_lungs * mask_nodule)
        # save masked nodes for this patient
        num_images = len(nodules)
        final_imgs = np.ndarray([num_images,1,512,512],dtype=np.float32)
        for i in range(num_images):
            final_imgs[i,0] = nodules[i]
        np.save(OUTPUT_FOLDER + patient + '_nodules.npy', final_imgs)
        