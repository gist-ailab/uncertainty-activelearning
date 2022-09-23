import torch
import random
import os
import torchvision.models as models

def std_convert(state_dict):
    state_dict2 = dict()
    for key1 in state_dict.keys():
        key1_list = key1.split('.')
        if(key1_list[0]=='features'):
            if len(key1_list) == 3:
                if key1_list[1] == '0':
                    key2 = 'conv1.weight'
                if key1_list[1] == '1':
                    key2 = 'bn1.'+key1_list[-1]
            elif len(key1_list) >= 5:
                key2 = 'layer'+f'{int(key1_list[1])-3}.'+'.'.join(key1_list[2:])
        state_dict2[key2] = state_dict[key1]
    return state_dict2

def std_convert2(state_dict):
    state_dict2 = std_convert(state_dict)
    state_dict3 = dict()
    for key in state_dict2.keys():
        key2 = 'encoder_q.net.'+key
        # key3 = 'ema.net.'+key
        state_dict3[key2] = state_dict2[key]
        # state_dict3[key3] = state_dict2[key]
    return state_dict3
