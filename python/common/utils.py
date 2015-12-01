'''
Created on Aug 25, 2015

@author: phil
'''
import os, glob
import numpy as np
from PIL import Image, ImageOps, ImageFilter
#from numpy import dtype




def createSampleIndex(lbl_image_list, image_names, patch_size=101, extend_border=True):
    """
    Runs through all given images in the list and creates the label index by recording
    all x,y locations in a separate list. Optionally extends the border by the patch size.
    """
    
    if patch_size == None or patch_size < 0:
        patch_size=0
    
    # store label indices in a dictionary
    lbl_idx = {}
    n_imgs = len(lbl_image_list)
    for img_idx in xrange(n_imgs):
        #print img_idx
        # get file path
        fp = lbl_image_list[img_idx]
        # get width and height of the image
        im = Image.open(fp, 'r')
        data = np.array(im)
        
        if extend_border:
            # extend the border to get more samples (note that the border must be extended in the source images as well)
            data = extend_image_border(data, (patch_size, patch_size), padding='zero')
            # DO NOT convert to a float array!!
        
        pad = int(np.floor(patch_size/2))
        height = data.shape[0]
        width = data.shape[1]
        for y in xrange(pad,height-pad):
            for x in xrange(pad,width-pad):
                # record frequency of all labels
                if not lbl_idx.has_key(data[y,x]):
                    # if label did not exist, create a new list
                    lbl_idx[data[y,x]] = [];
    
                # append the sample to the correct list with file path and x,y coordinates of the pixel
                lbl_idx[data[y,x]].append({ 'file':image_names[img_idx], 'x':x, 'y':y });
        
                
    labels = []
    # convert all keys to a list 
    for k_ in sorted(lbl_idx.keys()):
        print "adding label index %s to list" % k_
        labels.append(lbl_idx.get(k_))
        
    print "Collected %s labels from %s images" % (len(labels),n_imgs)
    
    # return the labels as list
    return labels

def extend_image_border(image, input_shape, padding='reflect_101'):
    """
    Extend the border of an image using a specified padding strategy.
    
    Parameters
    ----------
    image : numpy array (uint8)
        the image
        
    input_shape : tuple
        height and width of the input window
    
    
    returns 
        a uint8 numpy array
    """
    # compue the required padding
    pad = int(np.ceil(input_shape[0]/2.0))
    height = image.shape[0]
    width = image.shape[1]
    
    # extend black first
    image = ImageOps.expand(Image.fromarray(image), pad, fill='black')
    # convert back to numpy array
    image = np.asarray(image, dtype=np.uint8)
    image.setflags(write=True)
    
    # extend the border of the source image according to the input_shape
    #if padding == 'zero':
        # do nothing
    if padding == 'reflect_101':
        # extend image border and fill it with corresponding values        
        for b_y in xrange(pad,pad+height):
            # left border [0:pad)
            c_x = pad
            x_min = 1
            x_max = pad+1
            for b_x in xrange(x_min,x_max):
                image[b_y,c_x-b_x] = image[b_y,c_x+b_x]
                
            # right border (pad+width:width+(2*pad)]
            c_x = image.shape[1]-pad-2
            for b_x in xrange(x_min,x_max+1):
                image[b_y,c_x+b_x] = image[b_y,c_x-b_x]
                
       
        for b_x in xrange(pad,pad+width):           
            # top border [0:pad)
            c_y = pad
            y_min = 1
            y_max = pad+1
            for b_y in xrange(y_min,y_max):
                image[c_y-b_y,b_x] = image[c_y+b_y,b_x]
                
            # bottom border (pad+height:height+(2*pad)]
            c_y = image.shape[0]-pad-1
            for b_y in xrange(y_min,y_max):
                image[c_y+b_y,b_x] = image[c_y-b_y,b_x]
        
        #plt.imshow(image, cmap='gray')
        #plt.show()    
        
        # top left corner
        c_y = pad
        c_x = pad
        for b_y in xrange(1,pad):
            for b_x in xrange(1,pad+1):
                image[c_y-b_y,c_x-b_x] = image[c_y+b_y,c_x+b_x]
                
        # top right corner
        c_y = pad
        c_x = image.shape[1]-pad-2
        for b_y in xrange(1,pad+1):
            for b_x in xrange(1,pad+2):
                image[c_y-b_y,c_x+b_x] = image[c_y+b_y,c_x-b_x]
                
        # bottom right corner
        c_y = image.shape[0]-pad-1
        c_x = image.shape[1]-pad-2
        for b_y in xrange(1,pad+1):
            for b_x in xrange(1,pad+2):
                image[c_y+b_y,c_x+b_x] = image[c_y-b_y,c_x-b_x]
        
        # bottom left corner
        c_y = image.shape[0]-pad-1
        c_x = pad
        for b_y in xrange(1,pad+1):
            for b_x in xrange(1,pad+1):
                image[c_y+b_y,c_x-b_x] = image[c_y-b_y,c_x+b_x]

    return image


def gaussian_blur(image,sigma=0.5):
    """
    Blur an image with a given sigma and kernel_size
    
    Parameters
    ----------
    
    image : PIL image
        the image
        
    sigma : float
        how many sigmas should be blurred
    """
    im_blurred = image.filter(ImageFilter.GaussianBlur(radius=sigma));
    return im_blurred
    
def listFiles(directory, ext=None):
    """
    General approach to listing files and getting their full path
    out argument 1: list of all files with full paths
    out argument 2: list of all file names (without paths and extensions)
    """
    if ext==None:
        lst = glob.glob(directory)
    else:
        lst = glob.glob(os.path.join(directory, '*.' + ext))    
    
    print("File list:\n %s"%lst)
    files = []    
    
    # remove all directories
    for idx in range(len(lst)):
        if (os.path.isfile(os.path.join(directory,lst[idx]))):
            files.append(lst[idx])
    
    files.sort(); 
    filenames = []
    for idx in range(len(files)):
        filenames.append(os.path.basename(files[idx])); # all file names with extension
        #filenames.append(os.path.splitext(files[idx])[0])  # all file names (without extension)
        files[idx] = os.path.join(directory, files[idx]) # prepend the directory
  
    return files, filenames
    
def readLabelMap(filepath):
    label_map = {}
    
    if not os.path.exists(os.path.abspath(filepath)):
        raise "Cannot find the label file!"
    
    with open(filepath, 'rb') as f:
        for line in f:
            key, value = line.split()
            label_map[key] = value
        
    return label_map
    