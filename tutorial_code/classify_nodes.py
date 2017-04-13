# usage: python classify_nodes.py nodes.npy 

import numpy as np
import pandas as pd
import scipy as sp
import glob
import pickle
from tqdm import tqdm

from sklearn.cross_validation import StratifiedKFold as KFold
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from skimage.measure import regionprops, label

def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll

def getRegionFromMap(slice_npy):
    thr = np.where(slice_npy > np.mean(slice_npy),1.0,0.0)
    label_image = label(thr)
    labels = label_image.astype(int)
    regions = regionprops(labels)
    return regions

def getRegionMetricRow(fname = "nodules.npy"):
    # fname, numpy array of dimension [#slices, 1, 512, 512] containing the images
    stack = np.load(fname)
    nslices = stack.shape[0]
    
    #metrics
    totalArea = 0.
    avgArea = 0.
    maxArea = 0.
    avgEcc = 0.
    avgEquivlentDiameter = 0.
    stdEquivlentDiameter = 0.
    weightedX = 0.
    weightedY = 0.
    numNodes = 0.
    numNodesperSlice = 0.
    # crude hueristic to filter some bad segmentaitons
    # do not allow any nodes to be larger than 10% of the pixels to eliminate background regions
    maxAllowedArea = 0.10 * 512 * 512 
    
    areas = []
    eqDiameters = []
    for i in range(nslices):
        regions = getRegionFromMap(stack[i,0,:,:])
        for region in regions:
            if region.area > maxAllowedArea:
                continue
            totalArea += region.area
            areas.append(region.area)
            avgEcc += region.eccentricity
            avgEquivlentDiameter += region.equivalent_diameter
            eqDiameters.append(region.equivalent_diameter)
            weightedX += region.centroid[0]*region.area
            weightedY += region.centroid[1]*region.area
            numNodes += 1
            
    weightedX = weightedX / totalArea 
    weightedY = weightedY / totalArea
    avgArea = totalArea / numNodes
    avgEcc = avgEcc / numNodes
    avgEquivlentDiameter = avgEquivlentDiameter / numNodes
    stdEquivlentDiameter = np.std(eqDiameters)
    maxArea = max(areas)    
    numNodesperSlice = numNodes*1. / nslices
        
    return np.array([nslices,avgArea,maxArea,avgEcc,avgEquivlentDiameter,\
                     stdEquivlentDiameter, weightedX, weightedY, numNodes, numNodesperSlice])

def createTrainingDataset(nodfiles=None):
    if nodfiles == None:
        # directory of numpy arrays containing masks for nodules
        # found via unet segmentation
        noddir = "../data/segmented_nodules/stage1/" 
        nodfiles = glob.glob(noddir +"*_nodule_masks.npy")
    # dict with mapping between training examples and true labels
    # the training set is the output masks from the unet segmentation
    truthdata = pd.read_csv("../data/stage1/stage1_labels_all.csv")
    numfeatures = 10
    feature_array = np.zeros((len(nodfiles),numfeatures))
    truth_metric = np.zeros((len(nodfiles)))
    
    for i, nodfile in enumerate(tqdm(nodfiles)):
        patID = nodfile.split("\\")[1].split('_')[0]        
        truth_metric[i] = truthdata[truthdata['id']==patID]['cancer'].values[0]
        feature_array[i] = getRegionMetricRow(nodfile)
    
    np.save("dataY.npy", truth_metric)
    np.save("dataX.npy", feature_array)
    
def createTestDataset():
    noddir = "../data/segmented_nodules/test/" 
    nodfiles = glob.glob(noddir +"*_nodule_masks.npy")
    
    numfeatures = 10
    feature_array = np.zeros((len(nodfiles),numfeatures))
    patID_list = []
    
    for i, nodfile in enumerate(tqdm(nodfiles)):
        patID = nodfile.split("\\")[1].split('_')[0]
        feature_array[i] = getRegionMetricRow(nodfile)
        patID_list.append(patID)

    np.save("testX.npy", feature_array)
    pickle.dump( patID_list, open( "testIDs.p", "wb" ) )

def trainClassifier():
    X = np.load("dataX.npy")
    Y = np.load("dataY.npy")

    kf = KFold(Y, n_folds=3)
    y_pred = Y * 0
    for train, test in kf:
        X_train, X_test, y_train, y_test = X[train,:], X[test,:], Y[train], Y[test]
        clf = RandomForestClassifier(n_estimators=100, n_jobs=3)
        clf.fit(X_train, y_train)
        y_pred[test] = clf.predict(X_test)
    print(classification_report(Y, y_pred, target_names=["No Cancer", "Cancer"]))
    print("logloss",logloss(Y, y_pred))
    
    return clf

def evaluateTestData(classifier):
    # load test data
    X = np.load("testX.npy")
    patID_list = pickle.load( open( "testIDs.p", "rb" ) )
    
    # classify
    Y = clf.predict(X)
    
    # write
    d = {'id' : pd.Series(patID_list),
         'cancer' : pd.Series(Y)}
    df = pd.DataFrame(d)
    df = df[['id', 'cancer']]
    df.to_csv('submission.csv', index=False)

if __name__ == "__main__":
    from sys import argv  
    
#    createTrainingDataset()
    clf = trainClassifier()
    
    createTestDataset()
    evaluateTestData(clf)
