import torch

from MobileNetV3 import MobileNetV3


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


def mobilenetv3(num_classes=1000, scale=1., in_channels=3, small=False):
    model = MobileNetV3(num_classes=num_classes,
                        scale=scale,
                        in_channels=in_channels,
                        small=small)

    state = torch.load('mobilenetv3/results/mobilenetv3large-v1/model_best0-ec869f9b.pth')
    model.load_state_dict(prepare_state_dict(state['state_dict'], num_classes),
                          map_location='cpu')

    return model
