import os, multiprocessing
from os.path import join, isfile

import numpy as np
from PIL import Image
import scipy.ndimage
import skimage.color
#from sklearn import cluster

import sklearn
from sklearn import cluster
from scipy.spatial.distance import cdist
import opts


def extract_filter_responses(opts, img):
    '''
    Extracts the filter responses for the given image.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    [output]
    * filter_responses: numpy.ndarray of shape (H,W,3F)
    '''
    filter_scales = opts.filter_scales
    #filter_scales.append(3)
    
    # ----- TODO -----
    #r,g,b = img.split()
    
    try:
        labcolor = skimage.color.rgb2lab(img)
        if (labcolor.shape[2] >= 3):
            red = labcolor[...,0]
            green = labcolor[...,1]
            blue = labcolor[...,2]
    except:
        red = img
        green = img
        blue = img
    labcolor_sep = [red, green, blue]
    filter_responses = np.zeros([img.shape[0], img.shape[1], 3*4*len(filter_scales)])

    for i, scale in enumerate(filter_scales):
        for channel in range(len(labcolor_sep)):
            filter_responses[:,:,channel*4+12*i] = scipy.ndimage.gaussian_filter(labcolor_sep[channel], scale, mode = 'constant')
            filter_responses[:,:,channel*4+1+12*i] = scipy.ndimage.gaussian_laplace(labcolor_sep[channel], scale, mode = 'constant')
            filter_responses[:,:,channel*4+2+12*i] = scipy.ndimage.gaussian_filter(labcolor_sep[channel], scale, order=(0,1), mode = 'constant')
            filter_responses[:,:,channel*4+3+12*i] = scipy.ndimage.gaussian_filter(labcolor_sep[channel], scale, order = (1,0), mode = 'constant')


    
    return filter_responses


def compute_dictionary_one_image(args):
    '''
    Extracts a random subset of filter responses of an image and save it to disk
    This is a worker function called by compute_dictionary

    Your are free to make your own interface based on how you implement compute_dictionary
    '''

    # ----- TODO -----
    pass

def compute_dictionary(opts, n_worker=3):
    '''
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * opts         : options
    * n_worker     : number of workers to process in parallel
    
    [saved]
    * dictionary : numpy.ndarray of shape (K,3F)
    '''

    data_dir = opts.data_dir
    feat_dir = opts.feat_dir
    out_dir = opts.out_dir
    K = opts.K

    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
    # ----- TODO -----
    #pixels in each direction
    filter_scales = opts.filter_scales
    n = len(filter_scales)
    pi = opts.alpha//2
    #filter_responses = np.zeros([pi**2*len(train_files),3*4*3])
    for j in range(len(train_files)):
        img = Image.open(join(data_dir,train_files[j]))
        img = np.array(img).astype(np.float32)/255
        try:
            r,c, ch = img.shape
        except:
            r,c = img.shape

        #img = img[int(0.2*r):int(0.2*r+pi), int(0.2*c):int(0.2*c+pi)] + img[int(0.77*r):int(0.77*r+pi), int(0.2*c):int(0.2*c+pi)] + img[int(0.2*r):int(0.2*r+pi), int(0.77*c):int(0.77*c+pi)] + img[int(0.77*r):int(0.77*r+pi), int(0.77*c):int(0.77*c+pi)] + img[int(0.48*r):int(0.48*r+pi), int(0.48*c):int(0.48*c+pi)]
        #print("ith iteration", j)
        #print(img.shape)
        #print("pi x pi filter response", extract_filter_responses(opts, img).shape)
        filter_resp = extract_filter_responses(opts, img).reshape(r*c,3*4*n)
        rand_pixel = np.random.randint(r*c, size = opts.alpha)
        #filter_responses[j*pi**2:j*pi**2+pi**2,:] = extract_filter_responses(opts, img).reshape(pi**2, 3*4*3) #changed from filt_res
        sample_filter_response = filter_resp[rand_pixel,:]
        filter_responses = np.asarray(sample_filter_response)
    kmeans = sklearn.cluster.KMeans(n_clusters=K).fit(filter_responses)
    dictionary = kmeans.cluster_centers_
    

    ## example code snippet to save the dictionary
    np.save(join(out_dir, 'dictionary.npy'), dictionary)
    #return dictionary

def get_visual_words(opts, img, dictionary):
    '''
    Compute visual words mapping for the given img using the dictionary of visual words.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    
    [output]
    * wordmap: numpy.ndarray of shape (H,W)
    '''
    
    # ----- TODO -----
    try:
        ri,ci,chi = img.shape
    except:
        ri, ci = img.shape
    rd,cd = dictionary.shape
    #wordmap = np.zeros([ri,ci]).reshape(ri*ci,-1)
    filter_scales = opts.filter_scales
    n = len(filter_scales)
    filter_responses = extract_filter_responses(opts, img) #changed from filt_res
    modified_filter_response = filter_responses.reshape(ri*ci,3*4*n)
    dist = cdist(modified_filter_response, dictionary)
    wordmap = np.zeros(ri*ci)
    wordmap = np.argmin(dist, axis=1)
    
    #dist = scipy.spatial.distance.cdist(filter_responses, dictionary)
    #for i in range(ri*ci):
        #temp_min = min(dist[i,:])
        #col = np.where(dist[i,:] == temp_min)[0][0]
        #if col>9:
            #col = 9
        #val = np.mean(dictionary[col,:])
        #wordmap[i,:] = val

    wordmap = wordmap.reshape(ri,ci)
    #img_response = extract_filter_responses(opts, img)
    #H, W, F = img_response.shape
    #print("F", F)
    #img_response = img_response.reshape(-1,12)
    #dist = cdist(img_response, dictionary)
    #wordmap = np.argmax(dist, axis=1).reshape(H,W)

    return wordmap

