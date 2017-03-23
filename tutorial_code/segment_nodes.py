# -*- coding: utf-8 -*-

import numpy as np # linear algebra
import dicom
import os
from tqdm import tqdm

import LUNA_segment_lung_ROI
import LUNA_train_unet

INPUT_FOLDER = '../data/sample_images/'
OUTPUT_FOLDER = '../data/segmented_nodes/'

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
    unet.load_weights('./unet.hdf5')
    
    # segment
    for count, patient in enumerate(tqdm(patients)):
        slices = get_slices(INPUT_FOLDER + patient)
        nodules = []
        for s in slices:
            img = s.pixel_array
            # segment lungs
            mask_lungs = LUNA_segment_lung_ROI.get_lung_mask(img)
            img_lungs = LUNA_segment_lung_ROI.apply_mask_normalize(img, mask_lungs)[0]
            
            if img_lungs is None:
                pass
            else:            
                # segment nodules
                mask_nodule = unet.predict(img_lungs, verbose=0)[0]
                nodules.append(img_lungs * mask_nodule)
        # save masked nodes for this patient
        num_images = len(nodules)
        final_imgs = np.ndarray([num_images,1,512,512],dtype=np.float32)
        for i in range(num_images):
            final_imgs[i,0] = nodules[i]
        np.save(OUTPUT_FOLDER + patient + 'nodules.npy', final_imgs)
        