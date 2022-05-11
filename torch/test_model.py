#!/usr/bin/env python

import os
import numpy as np
from cerranet import cerranet
import utils as image
import sys
import torch
import torch.nn as nn
import torch.optim as optim

def main():
    model_dir = './models'

    print("[Info] Reading model architecture...")
    net = cerranet(pretrained=True)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01)

    print("[Info] Loading a sample image...")
    img = image.load_img('../images/20200809_207_119_L4_95152.tif', target_size=(256, 256))

    # scale pixels between 0 and 1
    X = np.transpose(image.img_to_array(img) / 255., (2, 0, 1))

    # inference
    net.eval()
    output = net(torch.from_numpy(np.array([X])))
    output = nn.functional.softmax(output, dim=1)
    output = output.data.numpy()

    # show results
    print('Position of maximum probability: {}'.format(output[0].argmax()))
    print('Maximum probability: {:.5f}'.format(output[0].max()))

    # sort top five predictions from softmax output
    print('\nTop probabilities and labels:')
    for label in output[0].argsort()[::-1]:
        print('{1}: {0:.5f}'.format(output[0][label], label))

if __name__ == '__main__':
    main()
