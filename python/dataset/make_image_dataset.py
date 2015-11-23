#!/usr/bin/env python
'''
Create a (sparse) dataset for libsvm from a set of images.

Test the script with the following terminal command:
    ./make_image_dataset.py ../test/ ../test/ds.libsvm --sparse True --resize 20 --scale False

Created on Nov 23, 2015

@author: phil
'''

import sys, os
from PIL.Image import BILINEAR
prj_root = os.path.dirname(os.path.abspath(__file__))
# add root directory to the python path
sys.path.insert(0, prj_root + '/..')

from PIL import Image
from common import utils
import numpy as np
import argparse

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
    
    return parser

# which_set = 'rename'
# patch_size = 101 # input size to the network
# n_channels = 1 # just coldecon_HE2_R
# # randomly draw n samples from each of the lists
# n_samples = 500 # train 5000, test 5000; separators: use n_samples*10 if only rotation: (~40,000 raw)
# n_classes = 2
# n_batch_files = 30 # the number of pkl files to split the data into
# extend_border = True
# augment_data = True
# clahe = True

# data augmentation techniques
#rotation_angles = [i for i in xrange(36,360,36)] # samples are transformed by these rotation angles
#flip_x = False # data augmentation by flipping around x axis
#flip_y = False # data augmentation by flipping around y axis
#flip_xy= False # data augmentation by flipping around x AND y axis

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


def processImage(path, label):
    """
    Load the image from the specified path and create the libsvm format.
    """
    src_image_raw = Image.open(path, 'r')
    #if extend_border:
        # extend the source image borders
    #    src_image_raw = utils.extend_image_border(np.asarray(src_image_raw, dtype=np.uint8), (patch_size, patch_size))
        
    #plt.imshow(src_image_raw)
    #plt.show()
    
    # size normalization of the image
    if not (args.resize == None):
        src_image_raw = src_image_raw.resize(size=(int(args.resize), int(args.resize)), resample=BILINEAR)
    
    # convert to writable numpy array
    src_image = np.asarray(src_image_raw, dtype=np.float32)
    src_image.setflags(write=True)
    
    # scale from 0-255 between 0 and 1
    if args.scale == True:
        src_image /= 255.
    
    # make the libsvm format
    line = im2libsvm(src_image, label)
    
    return line


def createDataset(sources,output,labels,sparse):
    """
    Create a dataset by vectorizing the images and writing them line by line to a txt-file.
    Each pixel is a feature and is thus stored in libsvm-format:
    
    [label] [index0:value0] [index1:value1] ... [indexN:valueN]
    
    """
    # delete the output file
    if os.path.exists(os.path.abspath(output)):
        os.remove(output)
    
    # open the output file
    output_file = open(output, 'wb')
    
    # first, list the source files
    fpaths_src, fnames_src = utils.listFiles(sources, 'png')
    
    # read the label file
    if not (labels == None):
        label_map = utils.readLabelMap(labels)
        # check that the numbers match
        assert len(label_map.keys()) == len(fpaths_src)
    
    # some dummy label
    label = -99.99
    
    # now process the images
    for img_idx in xrange(len(fpaths_src)):
        if not (labels == None):
            label = label_map[fnames_src[img_idx]]
        else:
            # add a dummy label
            label = np.random.rand()
            
        line = processImage(fpaths_src[img_idx], label)
        output_file.writelines(line + '\n')
    
    output_file.close()
    
    return 0;
    
if __name__ == "__main__":
    parser = make_argument_parser()
    args = parser.parse_args()
    ret = createDataset(args.sources, args.output, args.labels, args.sparse)
    print "Done."
    if not ret:
        sys.exit(-1)
