# -*- coding: utf-8 -*-

import numpy as np # linear algebra
import dicom
import os
from tqdm import tqdm
import warnings

import LUNA_segment_lung_ROI
import LUNA_train_unet

INPUT_FOLDER = '../data/sample_images/'
OUTPUT_FOLDER = '../data/segmented_nodules/'

# Load the scans in given folder path
def get_dicom_stack(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    return slices

def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)
    
    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):
        
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
            
        image[slice_number] += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)


if __name__ == "__main__":
    # find scan data
    patients = os.listdir(INPUT_FOLDER)
    patients.sort()
    print('-'*40)
    print('Prediciting node masks')
    print('-'*40)
    
    # load UNET nodule segmentation model
    unet = LUNA_train_unet.get_unet()
    unet.load_weights('./unet_double_trained.hdf5')
    
    # segment
    for i in range(len(patients)):
        print('Processing patient', str(i+1), 'of', str(len(patients)))
        patient = patients[i]
        stack = get_dicom_stack(INPUT_FOLDER + patient)
        slices = get_pixels_hu(stack)
        nodule_masks = []
        nodule_imgs = []
        lung_masks = []
        lung_imgs = []
        
        for count, img in enumerate(tqdm(slices)):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                # segment lungs
                mask_lungs = LUNA_segment_lung_ROI.get_lung_mask(img)
                if sum(mask_lungs.flatten()) is 0:
                    img_lungs = None
                else:
                    img_lungs = LUNA_segment_lung_ROI.apply_mask_normalize(img, mask_lungs)
                
                if img_lungs is not -1:
                    # segment nodules
                    mask_nodule = unet.predict(img_lungs.reshape([1,1,512,512]), verbose=0)[0]
                    
                    # TODO: write nodule count, sizes, slice number
                    
                    nodule_masks.append(mask_nodule)
                    nodule_imgs.append(mask_nodule* img_lungs)
                    lung_masks.append(mask_lungs)
                    lung_imgs.append(img_lungs)
        # save masked nodes for this patient
        num_images = len(nodule_imgs)
        final_nodule_imgs = np.ndarray([num_images,1,512,512],dtype=np.float32)
        final_nodule_masks = np.ndarray([num_images,1,512,512],dtype=np.float32)
        final_lung_imgs = np.ndarray([num_images,1,512,512],dtype=np.float32)
        final_lung_masks = np.ndarray([num_images,1,512,512],dtype=np.float32)
        for i in range(num_images):
            final_nodule_imgs[i,0] = nodule_imgs[i]
            final_nodule_masks[i,0] = nodule_masks[i]
            final_lung_imgs[i,0] = lung_imgs[i]
            final_lung_masks[i,0] = lung_masks[i]
        np.save(OUTPUT_FOLDER + patient + '_nodule_imgs.npy', final_nodule_imgs)
        np.save(OUTPUT_FOLDER + patient + '_nodule_masks.npy', final_nodule_masks)
        np.save(OUTPUT_FOLDER + patient + '_lung_imgs.npy', final_lung_imgs)
        np.save(OUTPUT_FOLDER + patient + '_lung_masks.npy', final_lung_masks)
        