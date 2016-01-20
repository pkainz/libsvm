#!/usr/bin/env python
'''
Create a bag-of-visual-words codebook from a set of image features (SIFT).
Each row in the codebook is a cluster center identified by k-means clustering.  

Run the script with the following terminal command:
    ./make_bow_codebook.py ../test/ ../test/test.bowcodebook --resize 32 --augment 1 --njobs 2 --features dsift --k 32

Created on Dec 10, 2015

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
        description="Make an bag of words codebook from a set of images"
    )
    parser.add_argument('sources',
                        help='Specifies the source image files/directory')
    parser.add_argument('output',
                        help='Specifies the output file path for the codebook')
    parser.add_argument('--resize',
                        default=None,
                        help='Specifies a spatial normalization width and height for each image')
    parser.add_argument('--augment',
                        default=0,
                        help='Specifies, whether to augment the image data with rotations and mirroring')
    parser.add_argument('--k',
                        default=32,
                        help='Specifies the size of the codebook (k clusters)')
    parser.add_argument('--njobs',
                        default=multiprocessing.cpu_count(),
                        help='Specifies the number of cpus for multiprocessing')
    parser.add_argument('--features',
                        default='dsift',
                        help='Specifies a list of features to be concatenated. \n' 
                            + 'dsift: dense SIFT descriptor for all channels in the image\n')
    
    return parser


def extractFeatures(image, feature_list):
    """
    Extracts a number of features for a channel. Features are concatenated into a 
    row-vector. 
    """
    # sift has 128D
    feat_vec = np.empty((0,128))
    n_channels = (image.shape[2] if len(image.shape)==3 else 1)
    
    img_f32 = image.astype(np.float32)

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
   
        else:
            raise Exception("Method '%s' is not implemented!"%(feature))
    
    return feat_vec

def processImage(fpaths_src, fnames_src, img_idx):
    """
    Load the image from the specified path and compute the features.
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
    
    image_features = []
    
    # add the original 
    image_features.append(extractFeatures(src_image, args.features))
    
    if args.augment == 1:
        print "Augmenting dataset..."
        # data augmentation techniques
        rotation_angles = [i for i in xrange(36,360,36)] # samples are transformed by these rotation angles
        
        flip_x = True # data augmentation by flipping around x axis
        flip_y = True # data augmentation by flipping around y axis
        flip_xy= True # data augmentation by flipping around x AND y axis
        
        for angle in rotation_angles:
            #print "Computing angle %s" % angle
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
            image_features.append(extractFeatures(rot_sample_crop, args.features))
            
            # add 3 flipped copies
            if flip_x:
                rot_sample_crop_x = cv2.flip(rot_sample_crop,0)
                image_features.append(extractFeatures(rot_sample_crop_x, args.features))
            if flip_y:
                rot_sample_crop_y = cv2.flip(rot_sample_crop,1)
                image_features.append(extractFeatures(rot_sample_crop_y, args.features))
            if flip_xy:
                rot_sample_crop_xy = cv2.flip(rot_sample_crop,-1)
                image_features.append(extractFeatures(rot_sample_crop_xy, args.features))
    
    counter+=1

    return np.concatenate(image_features, axis=0)



def createDataset(sources,output):
    """
    Create a codebook for a bag of visual words representation using k-means clustering
    
    """
    global has_joblib
    out_path = str(output)
    # delete the output file
    if os.path.exists(out_path):
        os.remove(out_path)
       
    # first, list the source files
    fpaths_src, fnames_src = utils.listFiles(directory=os.path.abspath(sources), ext='png')
         
    n_imgs = len(fpaths_src)
    
    all_features_list = []
    
    # parallel implementation (default, if joblib available)
    if has_joblib:
        image_features = Parallel(n_jobs=args.njobs,verbose=5) (delayed(processImage)(fpaths_src, fnames_src, img_idx) for img_idx in range(n_imgs))
        # stack the individual images
        image_features = np.concatenate(image_features,axis=0)
        #print image_features.shape
        all_features_list.append(image_features)
    else:
        for img_idx in xrange(n_imgs):
            image_features = processImage(fpaths_src, fnames_src, img_idx)
            image_features = np.concatenate(image_features,axis=0)
            all_features_list.append(image_features)
        
        
    # create k clusters from all features
    print "Clustering (k=%s)"%str(args.k)
    feat_matrix = np.concatenate(all_features_list, axis=0).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    _,labels,codebook = cv2.kmeans(
                                   feat_matrix,
                                   args.k,
                                   criteria,
                                   10,
                                   flags) 
        
    # write the codebook to the file using savetext() on the numpy array
    np.savetxt(out_path, 
               codebook, 
               delimiter=' ', 
               header=('Codebook, %s words, %s dimensions'%(str(args.k),str(feat_matrix.shape[1]))))
    
    return 0
    
if __name__ == "__main__":
    parser = make_argument_parser()
    args = parser.parse_args()
    # some type conversion
    args.augment = int(args.augment)
    args.njobs = int(args.njobs)
    args.k = int(args.k)
    args.features = args.features.split(',')

    # global counter
    counter = 0
    ret = createDataset(args.sources, args.output)
    print "Done."
    if not ret:
        sys.exit(-1)
