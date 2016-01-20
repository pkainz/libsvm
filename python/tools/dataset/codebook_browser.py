#!/usr/bin/env python
"""
codebook browser, launched with
    ./codebook_browser.py ../test/test.bowcodebook
"""
import sys
import argparse
import numpy as np
from matplotlib import pyplot as plt

def make_argument_parser():
    """
    Creates an ArgumentParser to read the options for this script from
    sys.argv
    """
    parser = argparse.ArgumentParser(
        description="Visualize a data file"
    )
    parser.add_argument('codebook_filename',
                        help='Specifies the codebook (numpy) file')
    parser.add_argument('--type',
                        default='histo',
                        help='Specifies the codebook (numpy) type, image or histo')
    parser.add_argument('--input_size',
                        default=49,
                        help='Size of the patches')
    parser.add_argument('--n_channels',
                        default=3,
                        help='Number of channels')
    return parser

def loadCodebook(codebook_file):
    return np.loadtxt(codebook_file, delimiter=' ', dtype=np.float32)

def visualize(file_path,type_,input_size,n_channels):
    
    global curr_idx
    # load the file 
    codebook = loadCodebook(file_path)
    # each line in the file is a codeword
    n_words = codebook.shape[0]
    n_dims = codebook.shape[1]
    print "Codebook contains %s words, %s dims each"%(n_words,n_dims)
    
    # make the figure window
    figure,axes = plt.subplots(1,1)
        
    def redraw():
        '''
        Draws the currently selected word.
        '''
        plt.cla()
        print type_
        if type_ == 'image':
            word = codebook[curr_idx,]
            if (n_channels == 1):
                word = np.reshape(word, (input_size,input_size), order='C')
            elif (n_channels == 3):
                word = np.reshape(word, (input_size,input_size,n_channels), order='C')
            
            if (n_channels == 1):
                axes.imshow(word, cmap='gray', interpolation='nearest')
            else:
                axes.imshow(word, interpolation='nearest')
        elif type_ == 'histo':
            word = codebook[curr_idx,]
            index = np.arange(n_dims)
            axes.bar(index, word, 1)
        else:
            raise Exception("Unknown codebook type, must be 'histo' or 'image'!")

        axes.set_title("Word #%i"%(curr_idx))
        figure.canvas.draw()
        
    
    def on_key_press(event):
        "Callback for key press events"
        global curr_idx
        if event.key in ('right'):
            if curr_idx >= n_words:
                curr_idx = n_words-1
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
    args.n_channels = int(args.n_channels)
    args.input_size = int(args.input_size)
    print args
    
    curr_idx = 0
    ret = visualize(args.codebook_filename,args.type,args.input_size,args.n_channels);
    
    if not ret:
        sys.exit(-1)
