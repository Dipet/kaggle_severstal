import os
import torch

from MobileNetV3 import mobilenetv3


def prepare_state_dict(state_dict, n_class):
    result = {}
    for key, item in state_dict.items():
        key = key.replace('module.', '')
        result[key] = item

        if key.startswith('last_block.fc.weight'):
            result[key] = torch.rand([n_class] + [item.size(1)])
        elif key.startswith('last_block.fc.bias'):
            result[key] = torch.zeros([n_class])
    return result
