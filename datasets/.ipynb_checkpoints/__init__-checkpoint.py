# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from .utils.continual_dataset import ContinualDataset
from .seq_cifar10 import SequentialCIFAR10
from .seq_cifar100 import SequentialCIFAR100
from .cifair10 import MyciFAIR10
from .cinic10 import CINIC10
from argparse import Namespace

NAMES = {
    SequentialCIFAR10.NAME: SequentialCIFAR10,
    SequentialCIFAR100.NAME: SequentialCIFAR100,
    MyciFAIR10.NAME:MyciFAIR10,
    CINIC10.NAME:CINIC10
}



def get_dataset(args: Namespace) -> ContinualDataset:
    """
    Creates and returns a continual dataset.
    :param args: the arguments which contains the hyperparameters
    :return: the continual dataset
    """
    assert args.dataset in NAMES.keys()
    return NAMES[args.dataset](args)





