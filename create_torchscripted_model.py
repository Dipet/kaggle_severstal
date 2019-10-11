import torch
import segmentation_models_pytorch as smp
import numpy as np
from pathlib import Path
from utils.mobilenetv3 import mobilenetv3


ckpt_path = Path("/mnt/HDD/home/druzhinin/kaggle/kaggle_severstal/download/11resnet50-hard-severstal/best.pth")
input = torch.from_numpy(np.random.random([2, 3, 256, 1600])).float()

device = torch.device("cpu")
model = smp.Unet("resnet50", encoder_weights=None, classes=4, activation=None)
model.to(device)
model.eval()
model.load_state_dict(torch.load(ckpt_path)['model_state_dict'], strict=True)

def set_requires_grad(model, requires_grad: bool):
    """
    Sets the ``requires_grad`` value for all model parameters.

    Args:
        model (torch.nn.Module): Model
        requires_grad (bool): value

    Examples:
        >>> model = SimpleModel()
        >>> set_requires_grad(model, requires_grad=True)
    """
    requires_grad = bool(requires_grad)
    for param in model.parameters():
        param.requires_grad = requires_grad

set_requires_grad(model, False)

module = torch.jit.trace(model.forward, input)

res = module(input)
print(res.shape)

torch.jit.save(module, str(ckpt_path.parent.joinpath('torchscript.pth')))