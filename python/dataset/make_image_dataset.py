#!/usr/bin/env python
'''
Create a (sparse) dataset for libsvm from a set of images.
The standard feature used is dense SIFT [Lowe, 1999], and if a SIFT codebook is available, the 
bag-of-visual-words approach is produced. The output will either be a nx128 feature matrix, or
an nxk matrix, where k is the number of visual codewords in the codebook. 
The labels in labels.txt are expected to start at 0 and to be continuous. 
This script then adds +1 to all labels, such that unlabeled instances are identified by random 
labels between [0,1).

If a positive label using 'ova' (one-versus-all) is specified, it corresponds to the label in 
your original data. 

Run the script with the following terminal command:
    ./make_image_dataset.py ../test/ ../test/ds.libsvm --sparse 1 --resize 32 --scale 1 --augment 1 --njobs 2 --labels ../test/labels.txt --features dsift --ova 6 --codebook ../test/test.bowcodebook

Created on Nov 23, 2015

@author: phil
'''

import sys, os
import multiprocessing
prj_root = os.path.dirname(os.path.abspath(__file__))
# add root directory to the python path
sys.path.insert(0, prj_root + '/..')

import cv2

from PIL import Image
from common import utils
import numpy as np
import argparse
try:
    from joblib import Parallel, delayed
    has_joblib = True
except ImportError:
    print("joblib is not available, running single thread")
    has_joblib = False

def make_argument_parser():
    """
    Creates an ArgumentParser to read the options for this script from
    sys.argv
    """
    parser = argparse.ArgumentParser(
        description="Make an libsvm-formatted dataset from a set of images"
    )
    parser.add_argument('sources',
                        help='Specifies the source image files/directory')
    parser.add_argument('output',
                        help='Specifies the libsvm-formatted output file path')
    parser.add_argument('--labels',
                        default=None,
                        help='Specifies a file defining a mapping from filename to ground truth label')
    parser.add_argument('--resize',
                        default=None,
                        help='Specifies a spatial normalization width and height for each image')
    parser.add_argument('--scale',
                        default=1,
                        help='Specifies, whether to (i) normalize features to zero-mean and unit-variance,'
                        + ' and then (ii) linearly scale all feature values between -1 and 1')
    parser.add_argument('--sparse',
                        default=1,
                        help='Specifies, whether to make sparse libsvm-format')
    parser.add_argument('--augment',
                        default=0,
                        help='Specifies, whether to augment the image data with rotations and mirroring')
    parser.add_argument('--njobs',
                        default=multiprocessing.cpu_count(),
                        help='Specifies the number of cpus for multiprocessing')
    parser.add_argument('--ova',
                        default=None,
                        help='Specifies the label ID, that will be the positive class')
    parser.add_argument('--codebook',
                        default=None,
                        help='Specifies the bag of words codebook to be used in the quantization for dsift features')
    parser.add_argument('--features',
                        default='dsift',
                        help='Specifies a list of features to be concatenated. \n' 
                            #+ 'color: vectorized intensities (for either grey value image or color image)\n' 
                            + 'dsift: dense SIFT descriptor for all channels in the image\n')
    
    return parser

def getKNNClassifier():
    """
    Load the BoW codebook and construct the kd search tree. 
    Compute the nearest neighbor out of 3 for a 'new_data' sample by calling 
        knn.find_nearest(new_data,3)
    """
    codebook = np.loadtxt(args.codebook, delimiter=' ', dtype=np.float32)
    
    args.nVisualWords = codebook.shape[0]
    
    # find nearest neighbor in the codebook
    knn = cv2.KNearest()
    # construct kd-tree with labels from 0 - (nCodewords-1)
    knn.train(codebook,np.arange(args.nVisualWords))
    
    return knn

def extractFeatures(image, feature_list):
    """
    Extracts a number of features for a channel. Features are concatenated into a 
    row-vector. 
    """
    # for multiple features  or color features
    #feat_vec = np.array([])
    
    # sift has 128D
    feat_vec = np.empty((0,128))
    n_channels = (image.shape[2] if len(image.shape)==3 else 1)
    
    #img_f32 = image.astype(np.float32)

    for feature in feature_list:
        if (feature.strip().lower() == 'dsift'):
            print "computing dsift (dense rootSift) features"
            dense = cv2.FeatureDetector_create("Dense")
            sift = cv2.SIFT()
            if n_channels == 1:
                kp = dense.detect(image[:,:])
                # compute kp descriptors
                _,des = sift.compute(image[:,:],kp)
                
                # normalize the descriptors (L1)
                des /= (des.sum(axis=1, keepdims=True) + 1e-7)
                des = np.sqrt(des)
  
                feat_vec = np.vstack((feat_vec, des))
            else:
                for channel in xrange(n_channels):
                    kp = dense.detect(image[:,:,channel])
                    _,des = sift.compute(image[:,:,channel],kp)
                    
                    # normalize the descriptors (L1)
                    des /= (des.sum(axis=1, keepdims=True) + 1e-7)
                    des = np.sqrt(des)

                    feat_vec = np.vstack((feat_vec, des))
        
#         if (feature.strip().lower() == 'color'):
#             print "computing color features"
#             # scale from 0-255 between 0 and 1
#             if args.scale == 1:
#                 img_f32 /= 255.
#             
#             f_tmp = img_f32.flatten()
#             feat_vec = np.append(feat_vec, f_tmp)
        else:
            raise Exception("Method '%s' is not implemented!"%(feature))  
    
    return feat_vec

def convert2libsvm(f_vec):
    """
    Create a sparse string representation of the feature vector.
    """
    line = ''
    
    label = int(f_vec[0])
    
    # if you specified a positive label ID in your data for one-versus-all
    if not (args.ova == None):
        line += '+1' if (str(label-1) == str(args.ova)) else '-1'
    else:
        line += str(label)
   
    # skip zero entries
    for feat_idx in xrange(2,f_vec.size+1):
        value = f_vec[feat_idx-1]
        if args.sparse == 1 and value == 0:
            continue
        else:
            line += (' ' + str(feat_idx) + ':' + str(f_vec[feat_idx-1]))
        
    return line

def getHistogramOfVisualWords(f_vec, knn):
    """
    Compute the nearest cluster center and return a histogram of codewords.
    """
    hist = np.zeros((1,args.nVisualWords)).flatten()
    
    print "Computing BoW histogram..."
    ret, results, neighbours, dist = knn.find_nearest(f_vec.astype(np.float32), 3)

    # count unique values
    y = np.bincount(results.astype(np.int64).flatten())
    ii = np.nonzero(y)[0]
    tmp = zip(ii,y[ii])

    # count the frequencies of the result labels
    for tuple_ in tmp:
        hist[tuple_[0]] = tuple_[1]
    
    return hist

def generateFeatures(src_image, label, knn=None):
    """
    Creates a single line for the dataset.
    """
    
    # computes the features
    f_vec = extractFeatures(src_image, args.features)
    
    # quantize, if codebook is present
    if not (knn == None):
        f_vec = getHistogramOfVisualWords(f_vec, knn)
    else:
        # flatten the array
        f_vec = f_vec.flatten()
    
    # prepend the label
    f_vec = np.hstack((label, f_vec))
    
    return f_vec

def processImage(fpaths_src, label_map, fnames_src, img_idx):
    """
    Load the image from the specified path and create the libsvm format.
    """
    global counter
    
    n_imgs = len(fpaths_src)
    print("Processing %s -- %s/%s (%s%%)"%(fnames_src[img_idx],counter,n_imgs,round(100.*counter/n_imgs)))
    
    path = fpaths_src[img_idx]
    src_image_raw = Image.open(path, 'r')
        
    # size normalization of the image
    if not (args.resize == None):
        src_image_raw = src_image_raw.resize(size=(int(args.resize), int(args.resize)), resample=Image.BILINEAR)
    
    # convert to writable numpy array
    src_image = np.asarray(src_image_raw, dtype=np.uint8)
    src_image.setflags(write=True)
    
    # some dummy label
    label = -99.99
    # the labels
    if not (label_map == {}):
        # let the label start at 1, instead of 0
        label = int(label_map[fnames_src[img_idx]])+1
    else:
        # add a dummy label (between 0 and 1)
        label = np.random.rand()
    
    image_features = []
    
    # add the original
    image_features.append(generateFeatures(src_image,label,args.knn))
    
    if args.augment == 1:
        print "Augmenting dataset..."
        # data augmentation techniques
        rotation_angles = [i for i in xrange(36,360,36)] # samples are transformed by these rotation angles
        
        flip_x = True # data augmentation by flipping around x axis
        flip_y = True # data augmentation by flipping around y axis
        flip_xy= True # data augmentation by flipping around x AND y axis
        
        for angle in rotation_angles:
            rot_matrix = cv2.getRotationMatrix2D(
                                                 (src_image.shape[1]/2.,src_image.shape[0]/2.),
                                                 angle,
                                                 1.0)
            rot_sample_crop = np.array([])
            rot_sample_crop = cv2.warpAffine(src_image,
                           rot_matrix,
                           (src_image.shape[1],src_image.shape[0]),
                           rot_sample_crop,
                           cv2.INTER_LINEAR,
                           cv2.BORDER_REFLECT_101)
            
            # add the sample to the dataset
            image_features.append(generateFeatures(rot_sample_crop,label,args.knn))
            
            # add 3 flipped copies
            if flip_x:
                rot_sample_crop_x = cv2.flip(rot_sample_crop,0)
                image_features.append(generateFeatures(rot_sample_crop_x,label,args.knn))
            if flip_y:
                rot_sample_crop_y = cv2.flip(rot_sample_crop,1)
                image_features.append(generateFeatures(rot_sample_crop_y,label,args.knn))
            if flip_xy:
                rot_sample_crop_xy = cv2.flip(rot_sample_crop,-1)
                image_features.append(generateFeatures(rot_sample_crop_xy,label,args.knn))
    
    counter+=1

    # return a nx128 or nxk matrix for the features of all modifications of this image
    return np.concatenate(image_features, axis=0)



def createDataset(sources,output,labels,sparse):
    """
    Create a dataset by vectorizing the images and writing them line by line to a txt-file.
    Each pixel is a feature and is thus stored in libsvm-format:
    
    [label] [index0:value0] [index1:value1] ... [indexN:valueN]
    
    """
    global has_joblib
    out_path = str(output)
    # delete the output file
    if os.path.exists(os.path.abspath(out_path)):
        os.remove(os.path.abspath(out_path))
    
    # first, list the source files
    fpaths_src, fnames_src = utils.listFiles(directory=os.path.abspath(sources), ext='png')
    
    label_map={}
    
    # read the label file
    if not (labels == None):
        label_map = utils.readLabelMap(labels)
        # check that the numbers match
        print("Number of images in label map : %s"%str(len(label_map.keys())-1))
        print("Number of images in source dir: %s"%str(len(fpaths_src)))
        assert len(label_map.keys())-1 == len(fpaths_src)
    
    # generate KNN classifier
    if not (args.codebook == 'None' or args.codebook == None):
        args.knn = getKNNClassifier()  
    else:
        args.knn = None
    
    # precompute number of images
    n_imgs = len(fpaths_src)
    
    # preallocate array
    # if augmentation, calculate (9*4+1)*n samples
    all_features_list = []
        
    # parallel implementation (default, if joblib available)
    if has_joblib:
        image_features = Parallel(n_jobs=args.njobs,verbose=5) (delayed(processImage)(fpaths_src, label_map, fnames_src, img_idx) for img_idx in range(n_imgs))
        #image_features = np.concatenate(image_features,axis=0)
        all_features_list.append(image_features)
    else:
        for img_idx in xrange(n_imgs):
            image_features = processImage(fpaths_src, label_map, fnames_src, img_idx)
            all_features_list.append(image_features)
    
    # make a 2D matrix from the list of features (stack all images vertically)
    feat_matrix = np.concatenate(all_features_list, axis=0).astype(np.float32)    
    
    # do feature scaling # TODO BUG in scaling
    #if False:
    if (args.scale == 1):
        # preserve the labels
        label_vec = feat_matrix[:,0]
        feat_matrix = np.delete(feat_matrix,0,1)
        
        for feat_idx in xrange(feat_matrix.shape[1]):
            feat_vec = feat_matrix[:,feat_idx]
             
            # first shift to zero mean and unit variance
            #feat_vec_zeromean = feat_vec - feat_vec.mean()
            #feat_vec_unitvar = feat_vec_zeromean / (feat_vec.std() + 1e-10)
         
            # standardize/normalize between 0 and 1
            #feat_vec_std = (feat_vec - np.min(feat_vec)) / (np.max(feat_vec) - np.min(feat_vec) + 1e-10)
             
            #print feat_vec
             
            # then linearly scale between -1 and 1 
            feat_vec_scale = 1.0*feat_vec * (1 - -1) - 1        
         
            # set column back to matrix
            feat_matrix[:,feat_idx] = feat_vec_scale
        
        # finally add the label_vec again
        feat_matrix = np.concatenate((np.reshape(label_vec,(feat_matrix.shape[0],1)),feat_matrix), axis=1)
           
    else:
        print "Data may not be properly scaled, use the 'svm-scale' implementation."

    # open the output file
    output_file = open(os.path.abspath(out_path), 'wb')

    # run through the feature matrix    
    for row_idx in xrange(feat_matrix.shape[0]):
        # make the libsvm format
        line = convert2libsvm(feat_matrix[row_idx,:])
        output_file.writelines(line + "\n")
    output_file.close()
    
    return 0
    
if __name__ == "__main__":
    parser = make_argument_parser()
    args = parser.parse_args()
    # some type conversion
    args.augment = int(args.augment)
    args.njobs = int(args.njobs)
    args.scale = int(args.scale)
    args.sparse = int(args.sparse)
    args.features = args.features.split(',')

    # global counter
    counter = 0
    ret = createDataset(args.sources, args.output, args.labels, args.sparse)
    print "Done."
    if not ret:
        sys.exit(-1)
