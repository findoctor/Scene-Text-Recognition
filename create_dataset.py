import os
import lmdb 
import cv2
import numpy as np
from scipy.io import loadmat
import os


def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.fromstring(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.iteritems():
            txn.put(k, v)

def createDataset(indicator, outputPath, imageDir, labelPath, lexiconList=None, checkValid=True):
    """
    Create LMDB dataset for CRNN training.
    ARGS:
        indicator     : use 'testdata' or 'traindata'
        outputPath    : LMDB output path
        imageDir      : image Directory
        labelPath     : label path to the .mat file
        lexiconList   : (optional) list of lexicon lists
        checkValid    : if true, check the validity of every image
    """
    annots = loadmat(labelPath)
    nSamples = len(annots[indicator][0])
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1
    for i in range(nSamples):
        imgName = str(annots[indicator][0][i][0][0])
        imgLabel = str(annots[indicator][0][i][1][0])
        imagePath = os.path.join(imageDir, imgName)
        
        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue
        with open(imagePath, 'r') as f:
            imageBin = f.read()
        if checkValid:
            if not checkImageIsValid(imageBin):
                print('%s is not a valid image' % imagePath)
                continue

        imageKey = 'image-%09d' % cnt
        labelKey = 'label-%09d' % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = imgLabel
        if lexiconList:
            lexiconKey = 'lexicon-%09d' % cnt
            cache[lexiconKey] = ' '.join(lexiconList[i])
        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt-1
    cache['num-samples'] = str(nSamples)
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)


if __name__ == '__main__':
    '''
    # build train dataset
    outputPath = 'Data/lmdb'
    imageDir = 'Data/IIIT5K'
    labelPath = 'Data/IIIT5K/traindata.mat'
    createDataset(outputPath, imageDir, labelPath)
    '''
    # build test dataset
    outputPath = 'Data/lmdb'
    imageDir = 'Data/IIIT5K'
    labelPath = 'Data/IIIT5K/traindata.mat'
    indicator = 'traindata'
    createDataset(indicator,outputPath, imageDir, labelPath)