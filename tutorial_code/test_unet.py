# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import LUNA_train_unet
working_path = "../../DSB3Tutorial/output_with_orig_img/"

def plot_image_and_mask(image, mask):
    plt.imshow(image)
    plt.imshow(mask)
    plt.imshow(image*mask)

def test_model(model, name, imgs, true_masks, save_masks=True):
    print('-'*40)
    print('Predicting masks using', name, 'model...')
    print('-'*40)
    num_test = len(imgs)
    pred_masks = np.ndarray([num_test,1,512,512],dtype=np.float32)
    print('generating masks')
    for i in tqdm(range(num_test)):
        pred_masks[i] = model.predict([imgs[i:i+1]], verbose=0)[0]
    mean = 0.0
    print('evaluating accuracy')
    for i in tqdm(range(num_test)):
        mean+=LUNA_train_unet.dice_coef_np(true_masks[i,0], pred_masks[i,0])
    mean/=num_test
    print("Mean Dice Coeff : ",mean)
    if save_masks:
        np.save(working_path+'pred_masks.npy', pred_masks)
#        np.save('true_masks.npy', true_masks)

# load model and test data
model = LUNA_train_unet.get_unet()
imgs_test = np.load(working_path+"testImages.npy").astype(np.float32)
masks_test_true = np.load(working_path+"testMasks.npy").astype(np.float32)

# use tutorial weights
#model.load_weights('./unet_original.hdf5')
#test_model(model, 'tutorial', imgs_test, masks_test_true)

# use trained weights
#model.load_weights('./unet_trained_preserve_range_170330.hdf5')
#test_model(model, 'trained', imgs_test, masks_test_true)

# use trained weights
model.load_weights('./unet_preserve_range_double_trained_170331.hdf5')
test_model(model, 'trained', imgs_test, masks_test_true, True)
