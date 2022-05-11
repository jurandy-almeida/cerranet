#!/usr/bin/env python

import pickle
import numpy as np

from cerranet import CerraNet
from keras import layers

def get_weight(layer):
    return layer.get_weights()[0]

def get_bias(layer):
    return layer.get_weights()[1]

def get_state_dict():
    model = CerraNet()
    state_dict = dict()
    for layer in model.layers:
        if isinstance(layer, layers.Conv2D):
            state_dict['features.' + layer.get_config()['name'] + '.weight'] = np.transpose(get_weight(layer), (3, 2, 0, 1))
            state_dict['features.' + layer.get_config()['name'] + '.bias'] = get_bias(layer)
        elif isinstance(layer, layers.Dense):
            if layer.get_config()['name'] == 'fc7':
                state_dict['classifier.' + layer.get_config()['name'] + '.weight'] = np.reshape(np.transpose(np.reshape(get_weight(layer), (2, 2, 256, 256)), (3, 2, 0, 1)), (256, 1024))
            else:
                state_dict['classifier.' + layer.get_config()['name'] + '.weight'] = np.transpose(get_weight(layer), (1, 0))
            state_dict['classifier.' + layer.get_config()['name'] + '.bias'] = get_bias(layer)
    return state_dict

def main():
    with open('cerranet-keras.pth', 'wb') as f:
        pickle.dump(get_state_dict(), f)

if __name__ == '__main__':
    main()
