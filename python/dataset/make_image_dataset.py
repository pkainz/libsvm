#!/usr/bin/env python
'''
Create a (sparse) dataset for libsvm from a set of images.

Test the script with the following terminal command:
    ./make_image_dataset.py ../test/ ../test/ds.libsvm --sparse True --resize 32 --scale True --augment True --njobs 2

Created on Nov 23, 2015

@author: phil
'''

import sys, os
prj_root = os.path.dirname(os.path.abspath(__file__))
# add root directory to the python path
sys.path.insert(0, prj_root + '/..')

import cv2
import multiprocessing

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
                        default=True,
                        help='Specifies, whether to scale values between 0 and 1')
    parser.add_argument('--sparse',
                        default=True,
                        help='Specifies, whether to make sparse libsvm-format')
    parser.add_argument('--augment',
                        default=True,
                        help='Specifies, whether to augment the image data with rotations and mirroring')
    parser.add_argument('--njobs',
                        default=multiprocessing.cpu_count(),
                        help='Specifies the number of cpus for multiprocessing')
    
    return parser


def im2libsvm(image, label):
    """
    Vectorize the image and create sparse version of that.
    """
    v_image = image.flatten()
    
    line = ''
    line += str(label)
    
    # skip zero entries
    for feat_idx in xrange(v_image.size):
        value = v_image[feat_idx]
        if args.sparse and value == 0:
            continue
        else:
            line += (' ' + str(feat_idx) + ':' + str(v_image[feat_idx]))
        
    return line

def addSample(src_image, label):
    """
    Creates a single line for the dataset.
    """
    # scale from 0-255 between 0 and 1
    if args.scale:
        src_image /= 255.
    
    # make the libsvm format
    line = im2libsvm(src_image, label)
    return line + "\n"

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
    src_image = np.asarray(src_image_raw, dtype=np.float32)
    src_image.setflags(write=True)
    
    # some dummy label
    label = -99.99
    # the labels
    if not (label_map == {}):
        label = label_map[fnames_src[img_idx]]
    else:
        # add a dummy label
        label = np.random.rand()
    
    lines = ''
    
    # add the original label
    lines+=addSample(src_image,label)
    
    if args.augment == True:
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
            lines+=addSample(rot_sample_crop,label)
            
            # add 3 flipped copies
            if flip_x:
                rot_sample_crop_x = cv2.flip(rot_sample_crop,0)
                lines+=addSample(rot_sample_crop_x,label)
            if flip_y:
                rot_sample_crop_y = cv2.flip(rot_sample_crop,1)
                lines+=addSample(rot_sample_crop_y,label)
            if flip_xy:
                rot_sample_crop_xy = cv2.flip(rot_sample_crop,-1)
                lines+=addSample(rot_sample_crop_xy,label)
    
    counter+=1

    return lines



def createDataset(sources,output,labels,sparse):
    """
    Create a dataset by vectorizing the images and writing them line by line to a txt-file.
    Each pixel is a feature and is thus stored in libsvm-format:
    
    [label] [index0:value0] [index1:value1] ... [indexN:valueN]
    
    """
    global has_joblib
    out_path = str(output)
    # delete the output file
    if os.path.exists(out_path):
        os.remove(out_path)
    
    # open the output file
    output_file = open(out_path, 'wb')
    
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
        
      
    n_imgs = len(fpaths_src)
    
    # parallel implementation (default, if joblib available)
    if has_joblib:
        lines_par = Parallel(n_jobs=args.njobs,verbose=5) (delayed(processImage)(fpaths_src, label_map, fnames_src, img_idx) for img_idx in range(n_imgs))
        output_file.writelines(lines_par)
    else:
        for img_idx in xrange(n_imgs):
            line = processImage(fpaths_src, label_map, fnames_src, img_idx)
            output_file.writelines(line)
        
    output_file.close()
    
    return 0
    
if __name__ == "__main__":
    parser = make_argument_parser()
    args = parser.parse_args()
    # global counter
    global counter 
    counter = 0
    ret = createDataset(args.sources, args.output, args.labels, args.sparse)
    print "Done."
    if not ret:
        sys.exit(-1)
