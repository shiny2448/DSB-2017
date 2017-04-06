# -*- coding: utf-8 -*-

import numpy as np # linear algebra
import SimpleITK as sitk
from glob import glob
from tqdm import tqdm
import warnings

import LUNA_segment_lung_ROI
import LUNA_train_unet

INPUT_FOLDER = '../data/luna16/subset0/'
OUTPUT_FOLDER = '../data/segmented_nodules_luna/'

if __name__ == "__main__":
    # find scan data
    patients = glob(INPUT_FOLDER+'*.mhd')
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
        scan_filename = patients[i]
        itk_img = sitk.ReadImage(scan_filename) 
        slices = sitk.GetArrayFromImage(itk_img)
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
                    
                    # skip slices with no identified nodules
                    if sum(mask_nodule.flatten()) is 0:
                        pass
                    else:                        
                        # TODO: write nodule count, sizes, slice number                    
                        nodule_masks.append(mask_nodule)
                        nodule_imgs.append(mask_nodule* img_lungs)
                        lung_masks.append(mask_lungs)
                        lung_imgs.append(img_lungs)
        # save masked nodes for this patient
        num_images = len(nodule_imgs)
        print('Found ', str(num_images), ' slices with nodules')
        final_nodule_imgs = np.ndarray([num_images,1,512,512],dtype=np.float32)
        final_nodule_masks = np.ndarray([num_images,1,512,512],dtype=np.float32)
        final_lung_imgs = np.ndarray([num_images,1,512,512],dtype=np.float32)
        final_lung_masks = np.ndarray([num_images,1,512,512],dtype=np.float32)
        for i in range(num_images):
            final_nodule_imgs[i,0] = nodule_imgs[i]
            final_nodule_masks[i,0] = nodule_masks[i]
            final_lung_imgs[i,0] = lung_imgs[i]
            final_lung_masks[i,0] = lung_masks[i]
        np.save(OUTPUT_FOLDER + str(i) + '_nodule_imgs.npy', final_nodule_imgs)
        np.save(OUTPUT_FOLDER + str(i) + '_nodule_masks.npy', final_nodule_masks)
        np.save(OUTPUT_FOLDER + str(i) + '_lung_imgs.npy', final_lung_imgs)
        np.save(OUTPUT_FOLDER + str(i) + '_lung_masks.npy', final_lung_masks)
        