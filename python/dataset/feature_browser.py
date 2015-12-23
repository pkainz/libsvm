#!/usr/bin/env python
"""
codebook browser, launched with
    ./feature_browser.py ../test/ds.libsvm
"""
import sys
import argparse
import numpy as np
from matplotlib import pyplot as plt
import svmutil

def make_argument_parser():
    """
    Creates an ArgumentParser to read the options for this script from
    sys.argv
    """
    parser = argparse.ArgumentParser(
        description="Visualize a data file"
    )
    parser.add_argument('dataset_filename',
                        help='Specifies the dataset file in libsvm format')
    parser.add_argument('--type',
                        default='histo',
                        help='Specifies the dataset type, currently only histo supported')

    return parser

def visualize(file_path,type_,input_size=0,n_channels=0):
    
    global curr_idx
    # load the file 
    labels, dataset = svmutil.svm_read_problem(file_path)
    # each line in the file is a sample
    n_samples = len(dataset)
    n_dims = len(dataset[1])
    print "Dataset contains %s samples, %s dims each"%(n_samples,n_dims)
    
    # make the figure window
    figure,axes = plt.subplots(1,1)
        
    def redraw():
        '''
        Draws the currently selected sample.
        '''
        plt.cla()
#         print type_
#         if type_ == 'image':
#             sample = dataset[curr_idx,]
#             if (n_channels == 1):
#                 sample = np.reshape(sample, (input_size,input_size), order='C')
#             elif (n_channels == 3):
#                 sample = np.reshape(sample, (input_size,input_size,n_channels), order='C')
#             
#             if (n_channels == 1):
#                 axes.imshow(sample, cmap='gray', interpolation='nearest')
#             else:
#                 axes.imshow(sample, interpolation='nearest')
        if type_ == 'histo':
            sample = dataset[curr_idx]
            axes.bar(sample.keys(), sample.values(), 1)
        else:
            raise Exception("Unknown dataset type, must be 'histo'!")

        # get the label
        label = labels[curr_idx]

        axes.set_title("Sample #%i, label: %s"%(curr_idx,label))
        figure.canvas.draw()
        
    
    def on_key_press(event):
        "Callback for key press events"
        global curr_idx
        if event.key in ('right'):
            if curr_idx >= n_samples:
                curr_idx = n_samples-1
            else:
                #move to next image
                curr_idx+=1
            redraw()
        elif event.key in ('left'):
            if curr_idx < 0:
                curr_idx = 0
            else:
                #move to previous image
                curr_idx-=1
            redraw()
        elif event.key == 'q':
            sys.exit(0)

    # visualize the first image
    figure.canvas.mpl_connect('key_press_event', on_key_press)
    # show first image
    redraw()
    plt.show()

    return True

if __name__ == "__main__":
    parser = make_argument_parser()
    args = parser.parse_args()
    print args
    
    curr_idx = 0
    ret = visualize(args.dataset_filename, args.type)
    
    if not ret:
        sys.exit(-1)
