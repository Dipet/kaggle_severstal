from catalyst.dl import registry

from catalyst.dl import SupervisedRunner as Runner
from .experiment import Experiment

from utils.callbacks import DiceCallback as MyDice, IouCallback as MyIOU
from .model import Model

registry.Callback(MyDice, name='MyDice')
registry.Callback(MyIOU, name='MyIOU')