from os.path import join

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import util
import visual_words
import visual_recog
from opts import get_opts

import sys
sys.argv=['']
del sys

def main():
    opts = get_opts()

    ## Q1.1
    img_path = join(opts.data_dir, 'aquarium/sun_aztvjgubyrgvirup.jpg')
    img = Image.open(img_path)
    img = np.array(img).astype(np.float32)/255
    filter_responses = visual_words.extract_filter_responses(opts, img)
    util.display_filter_responses(opts, filter_responses)
    print(filter_responses.shape)

    ## Q1.2
    n_cpu = util.get_num_CPU()
    visual_words.compute_dictionary(opts, n_worker=n_cpu)
    #print(visual_words.compute_dictionary(opts, n_worker=n_cpu))
    #print("Shape of dictionary",visual_words.compute_dictionary(opts, n_worker=n_cpu).shape)
    
    ## Q1.3
    img_path = join(opts.data_dir, 'kitchen/sun_aasmevtpkslccptd.jpg') 
    img = Image.open(img_path)
    img = np.array(img).astype(np.float32)/255
    dictionary = np.load(join(opts.out_dir, 'dictionary.npy'))
    #print("Shape of dictionary",dictionary.shape)
    wordmap = visual_words.get_visual_words(opts, img, dictionary)
    util.visualize_wordmap(wordmap)

    ## Q2.1-2.4
    n_cpu = util.get_num_CPU()
    hist = visual_recog.get_feature_from_wordmap(opts, wordmap)
    print("Shape of hist", hist.shape)
    plt.hist(wordmap.flatten(), bins=np.arange(0,opts.K+1))
    plt.show()
    hist_all = visual_recog.get_feature_from_wordmap_SPM(opts, wordmap)
    #print("hist_all shape", hist_all.shape)
    visual_recog.build_recognition_system(opts, n_worker=n_cpu)

    ## Q2.5
    n_cpu = util.get_num_CPU()
    conf, accuracy = visual_recog.evaluate_recognition_system(opts, n_worker=n_cpu)
    
    print(conf)
    print(accuracy)
    np.savetxt(join(opts.out_dir, 'confmat.csv'), conf, fmt='%d', delimiter=',')
    np.savetxt(join(opts.out_dir, 'accuracy.txt'), [accuracy], fmt='%g')


if __name__ == '__main__':
    main()
