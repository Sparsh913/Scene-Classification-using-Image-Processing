import os, math, multiprocessing
from os.path import join
from copy import copy

import numpy as np
from PIL import Image

import visual_words
from sklearn.metrics import confusion_matrix

#wordmap = visual_words.get_visual_words(opts,img)

def get_feature_from_wordmap(opts, wordmap):
    '''
    Compute histogram of visual words.
 
    
    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist: numpy.ndarray of shape (K)
    '''

    K = opts.K
    # ----- TODO -----
    hist, bins = np.histogram(wordmap.flatten(),bins=np.arange(0, K+1))
    hist = hist / np.sum(hist)
    return hist

def get_feature_from_wordmap_SPM(opts, wordmap):
    '''
    Compute histogram of visual words using spatial pyramid matching.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist_all: numpy.ndarray of shape (K*(4^L-1)/3)
    '''
        
    K = opts.K
    L = opts.L # Layers of the pyramid
    # ----- TODO -----
    r,c = wordmap.shape
    hist_list = np.array([])
    l = L
    while(l>=0):
        slice_r, slice_c = int(r/(2**l))+1, int(c/(2**l))+1
        for i in range(0,r,slice_r):
            for j in range(0,c,slice_c):
                slice_wordmap = wordmap[i:i+slice_r,j:j+slice_c]
                hist = get_feature_from_wordmap(opts,slice_wordmap)
                if (l>1):
                    hist *= 2**(l-L-1)
                else:
                    hist *= 2**(-L)
                hist_list = np.append(hist_list,hist)
        l -= 1
    #hist_all = sum(hist_list)
    if np.sum(hist_list) > 0:
        return hist_list/np.sum(hist_list)
    return hist_list
    
def get_image_feature(opts, img_path, dictionary):
    '''
    Extracts the spatial pyramid matching feature.

    [input]
    * opts      : options
    * img_path  : path of image file to read
    * dictionary: numpy.ndarray of shape (K, 3F)


    [output]
    * feature: numpy.ndarray of shape (K*(4^L-1)/3)
    '''

    # ----- TODO -----
    #K = opts.K
    img = Image.open(img_path)
    img = np.array(img).astype(np.float32)/255
    wordmap = visual_words.get_visual_words(opts, img, dictionary)
    hist_all = get_feature_from_wordmap_SPM(opts, wordmap)

    return hist_all

#def get_image_feature_pool(args):
    # This function is made to simplify the usage of Pool which can only send 1 argurement at a time
    #return get_image_feature(args[0],args[1],args[2])

def build_recognition_system(opts, n_worker=8):
    '''
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N,M)
    * labels: numpy.ndarray of shape (N)
    * dictionary: numpy.ndarray of shape (K,3F)
    * SPM_layer_num: number of spatial pyramid layers
    '''
    K = opts.K
    data_dir = opts.data_dir
    out_dir = opts.out_dir
    SPM_layer_num = opts.L

    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()[:]
    train_labels = np.loadtxt(join(data_dir, 'train_labels.txt'), np.int32)[:]
    dictionary = np.load(join(out_dir, 'dictionary.npy'))

    # ----- TODO -----
    K = opts.KK = opts.K
    img_len = len(train_files)
    img_list = np.arange(img_len)
    spm_op = int(K*(4**(SPM_layer_num+1)-1)/3)
    features = np.zeros(spm_op).reshape((1,spm_op))
    #para_list = list(zip(img_list, train_files, train_labels))
    pool = multiprocessing.Pool(n_worker)
    for i in range(img_len):
        img_path = train_files[i]
        img_path = join(data_dir,img_path)
        ft = get_image_feature(opts,img_path, dictionary)
        #print("ith iteration", i)
        #print("Shape of ft", ft.shape)
        #print("Shape of features", features.shape)
        features = np.vstack((features, ft))
    #features = pool.map(get_image_feature_redirector,[(opts, join(opts.data_dir,img_path) ,dictionary) for img_path in train_files])
    #pool.map(get_image_feature_redirector, para_list)
    #for i in range(len(train_files[:10])):
        #img_path = img_path = join(data_dir, train_files[i])
        #ft = get_image_feature(opts, img_path, dictionary)
        #print("buildfunc - ft", ft)
        #features.append(ft)
    # We return None if an image cannot be read
    #filtered_features = [feature for feature in features if feature is not None]
    #train_labels = [label for idx, label in enumerate(train_labels) if features[idx] is not None]
    #concatenated_features = np.vstack(filtered_features)
    #pool.close()
    #pool.join()

        # Saving trained_system.npz
    np.savez_compressed(join(out_dir, 'trained_system.npz'), dictionary = dictionary, features = features, labels = train_labels, SPM_layer_num = SPM_layer_num)
    print("buildfunc - features", features)
    

    ## example code snippet to save the learned system
    # np.savez_compressed(join(out_dir, 'trained_system.npz'),
    #     features=features,
    #     labels=train_labels,
    #     dictionary=dictionary,
    #     SPM_layer_num=SPM_layer_num,
    # )

def distance_to_set(word_hist, histograms):
    '''
    Compute similarity between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N,K)

    [output]
    * hist_dist: numpy.ndarray of shape (N)
    '''

    # ----- TODO -----
    #n = histograms.shape[0]
    #hist_minimum = np.array([])
    #hist_dist = np.array([])
    #for i in range(n):
        #np.append(hist_minimum, np.minimum(word_hist, histograms[i,:]))
        #np.append(hist_dist, np.sum(hist_minimum))
    #hist_dist = 1 - hist_dist
    #return hist_dist
    mins = np.minimum(histograms, word_hist)
    sums = np.sum(mins, axis=1)
    hist_dist = 1-sums
    return hist_dist
    
def evaluate_recognition_system(opts, n_worker=8):
    '''
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8,8)
    * accuracy: accuracy of the evaluated system
    '''

    data_dir = opts.data_dir
    out_dir = opts.out_dir

    trained_system = np.load(join(out_dir, 'trained_system.npz'))
    dictionary = trained_system['dictionary']

    # using the stored options in the trained system instead of opts.py
    test_opts = copy(opts)
    test_opts.K = dictionary.shape[0]
    test_opts.L = trained_system['SPM_layer_num']

    test_files = open(join(data_dir, 'test_files.txt')).read().splitlines()[:]
    test_labels = np.loadtxt(join(data_dir, 'test_labels.txt'), np.int32)[:]

    # ----- TODO -----
    features = trained_system['features']
    labels = trained_system['labels']
    #c_m = np.zeros((8,8)) # Confusion Matrix
    #for i in range(len(test_files[:10])):
        #img = Image.open(join(data_dir, test_files[i]))
        #img = np.array(img).astype(np.float32)/255
        #wordmap = visual_words.get_visual_words(opts, img, dictionary)
        #hist_spm = get_feature_from_wordmap_SPM(opts, wordmap)
        #print("hist_spm", hist_spm)
        #print('features', features)
        #dist = distance_to_set(hist_spm, features)
        #ground_truth_label = test_labels[i]
        #if len(dist) != 0: predict_label = labels[np.argmax(dist)] 
        #else: predict_label = 0
        #c_m[ground_truth_label, predict_label] += 1
        
    #accuracy = np.trace(c_m)/np.sum(c_m)

    # Working Code Start
    #pool = multiprocessing.Pool(processes=n_worker)   
    #test_features = pool.map(get_image_feature_pool,[(opts, join(opts.data_dir,img_path) ,dictionary) for img_path in test_files])
    
    #pool.close()
    #pool.join()
    
    #test_features = [feature for feature in test_features if feature is not None]
    #test_labels = [label for idx, label in enumerate(test_labels) if test_features[idx] is not None]
    
    #hist_dists = [distance_to_set(test_feature,features) for test_feature in test_features]
    #hist_dists = np.vstack(hist_dists)
    #min = np.argmin(hist_dists, axis=1)
    
    #correct_predictions = np.sum(labels[min] == test_labels)
    #total_predictions = len(test_labels)
    #acc = correct_predictions / total_predictions
    
    #c = confusion_matrix(labels[min], test_labels )
    # Working code ends

    # Trial
    #test_data  = np.asarray(test_files)
    predict = []
    count = 0
    for i in range(len(test_files)):
        img_path = test_files[i]
        img_path = join(data_dir,img_path)
        test_feature = get_image_feature(opts, img_path, dictionary)
        dist = distance_to_set(test_feature, features)
        idx = np.argmin(dist)
        count += 1
        predict.append(labels[idx])

    c = np.zeros((8,8))
    correct = 0
    for j in range(len(test_files)):
        r_c = predict[j]
        c_c = test_labels[j]
        c[r_c,c_c] += 1
        if r_c == c_c:
            correct += 1
        acc = correct/len(test_labels)
    return c, acc

