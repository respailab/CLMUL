""" ciFAIR data loaders for PyTorch.

Version: 1.0

https://cvjena.github.io/cifair/
"""

import torchvision.datasets
import torch
import torchvision.transforms as transforms
from backbone.ResNet18 import resnet18
import torch.nn.functional as F
from utils.conf import base_path
from PIL import Image
import numpy as np
from .utils.validation import get_train_val
from .utils.continual_dataset import ContinualDataset, store_loaders, store_masked_loaders
from .utils.continual_dataset import get_previous_train_loader
from typing import Tuple
from .transforms.denormalization import DeNormalize

class ciFAIR10(torchvision.datasets.CIFAR10):
    base_folder = 'ciFAIR-10'
    url = 'https://github.com/cvjena/cifair/releases/download/v1.0/ciFAIR-10.zip'
    filename = 'ciFAIR-10.zip'
    tgz_md5 = 'ca08fd390f0839693d3fc45c4e49585f'
    test_list = [
        ['test_batch', '01290e6b622a1977a000eff13650aca2'],
    ]
    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        super(ciFAIR10, self).__init__(root, train, transform, target_transform, download)

    def __getitem__(self, index: int) -> Tuple[type(Image), int, type(Image), int]:
        """
        Gets the requested element from the dataset.
        :param index: index of the element to be returned
        :returns: tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # to return a PIL Image
        img = Image.fromarray(img, mode='RGB')
        original_img = img.copy()

        not_aug_img = self.not_aug_transform(original_img)

        if self.transform is not None:
            img = self.transform(img)
        #print(img)
        #exit()
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, not_aug_img, 1

class MyciFAIR10(ContinualDataset):

    NAME = 'ciFAIR10'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 2
    N_TASKS = 5
    TRANSFORM = transforms.Compose(
            [transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465),
                                  (0.2470, 0.2435, 0.2615)),
              ])

    def get_data_loaders(self):
        transform = self.TRANSFORM

        test_transform = transforms.Compose(
            [transforms.ToTensor(), self.get_normalization_transform()])

        train_dataset = ciFAIR10(base_path() + 'ciFAIR-10', train=True,
                                  download=True, transform=transform)
        if self.args.validation:
            train_dataset, test_dataset = get_train_val(train_dataset,
                                                    test_transform, self.NAME)
        else:
            test_dataset = ciFAIR10(base_path() + 'ciFAIR-10', train=False,
                                   download=False, transform=test_transform)

        #train, test = store_loaders(train_dataset, test_dataset, self)
        train, test = store_masked_loaders(train_dataset, test_dataset, self)
        return train, test

    def not_aug_dataloader(self, batch_size):
        transform = transforms.Compose([transforms.ToTensor(), self.get_normalization_transform()])

        train_dataset = ciFAIR10(base_path() + 'ciFAIR-10', train=True,
                                  download=False, transform=transform)
        train_loader = get_previous_train_loader(train_dataset, batch_size, self)

        return train_loader

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), MyciFAIR10.TRANSFORM])
        return transform

    @staticmethod
    def get_backbone():
        return resnet18(MyciFAIR10.N_CLASSES_PER_TASK
                        * MyciFAIR10.N_TASKS)

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                         (0.2470, 0.2435, 0.2615))
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize((0.4914, 0.4822, 0.4465),
                                (0.2470, 0.2435, 0.2615))
        return transform
    

    