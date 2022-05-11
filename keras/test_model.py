#!/usr/bin/env python

import os
import numpy as np
from cerranet import CerraNet
from keras.preprocessing import image
import sys
import keras
import keras.backend as K
if keras.__version__ < '2.0.0':
    dim_ordering = K.image_dim_ordering()
else:
    dim_ordering = K.image_data_format()
print("[Info] image_dim_order (from default ~/.keras/keras.json)={}".format(
        dim_ordering))
backend = dim_ordering

def main():
    model_dir = './models'
    global backend

    # override backend if provided as an input arg
    if len(sys.argv) > 1:
        if 'tf' in sys.argv[1].lower():
            backend = 'tf'
        else:
            backend = 'th'
    print("[Info] Using backend={}".format(backend))

    print("[Info] Reading model architecture...")
    model = CerraNet()
    model.compile(loss='mean_squared_error', optimizer='sgd')

    print("[Info] Loading a sample image...")
    img = image.load_img('../images/20200809_207_119_L4_95152.tif', target_size=(256, 256))

    # scale pixels between 0 and 1 
    X = image.img_to_array(img) / 255.

    # inference
    output = model.predict_on_batch(np.array([X]))

    # show results
    print('Position of maximum probability: {}'.format(output[0].argmax()))
    print('Maximum probability: {:.5f}'.format(output[0].max()))

    # sort predictions from softmax output
    print('\nTop probabilities and labels:')
    for label in output[0].argsort()[::-1]:
        print('{1}: {0:.5f}'.format(output[0][label], label))

if __name__ == '__main__':
    main()
