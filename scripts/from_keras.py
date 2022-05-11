#!/usr/bin/env python

import torch
import pickle

from cerranet import CerraNet

def main():
    with open('cerranet-keras.pth', 'rb') as f:
        keras_state_dict = pickle.load(f)
    model = CerraNet()
    torch_state_dict = model.state_dict()
    for key in torch_state_dict.keys():
        torch_state_dict[key] = torch.from_numpy(keras_state_dict[key])
    model.load_state_dict(torch_state_dict)
    torch.save(model.state_dict(), 'cerranet-torch.pth')  

if __name__ == '__main__':
    main()
