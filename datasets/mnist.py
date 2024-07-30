import torch
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from backbone.ResNet18 import resnet18
import torch.nn.functional as F
from utils.conf import base_path
from PIL import Image
import numpy as np
from datasets.utils.validation import get_train_val
from datasets.utils.continual_dataset import ContinualDataset, store_masked_loaders, get_previous_train_loader
from typing import Tuple
from datasets.transforms.denormalization import DeNormalize


class MyMNIST(MNIST):
    """
    Overrides the MNIST dataset to adjust for grayscale images.
    """
    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:
        super(MyMNIST, self).__init__(root, train, transform, target_transform, download)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """
        Gets the requested element from the dataset.
        :param index: index of the element to be returned
        :returns: tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # Convert numpy array to PIL Image
        img = Image.fromarray(np.uint8(img), mode='L')

        if self.transform is not None:
            img_aug = self.transform(img)
        else:
            img_aug = transforms.ToTensor()(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.train:
            if self.transform is not None:
                img_not_aug = self.transform(img)
            else:
                img_not_aug = transforms.ToTensor()(img)
        else:
            img_not_aug = img_aug

        # Ensure consistency in image dimensions
        if img_aug.size != (1, 32, 32):
            img_aug = transforms.Resize((32, 32))(img_aug)
        if img_not_aug.size != (1, 32, 32):
            img_not_aug = transforms.Resize((32, 32))(img_not_aug)

        return img_aug, target, img_not_aug, 1


class SequentialMNIST(ContinualDataset):

    NAME = 'mnist'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 2
    N_TASKS = 5
    TRANSFORM = transforms.Compose(
        [transforms.RandomCrop(32, padding=4),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,)),
        ])

    def get_data_loaders(self):
        transform = self.TRANSFORM

        test_transform = transforms.Compose(
            [transforms.Resize((32, 32)), transforms.ToTensor(), self.get_normalization_transform()])

        train_dataset = MyMNIST(base_path() + 'MNIST', train=True,
                                download=True)
        train_dataset.transform = transform
        if self.args.validation:
            train_dataset, test_dataset = get_train_val(train_dataset,
                                                        test_transform, self.NAME)
        else:
            test_dataset = MyMNIST(base_path() + 'MNIST', train=False,
                                   download=False, transform=test_transform)

        train, test = store_masked_loaders(train_dataset, test_dataset, self)
        return train, test

    def not_aug_dataloader(self, batch_size):
        transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(), self.get_normalization_transform()])

        train_dataset = MyMNIST(base_path() + 'MNIST', train=True,
                                download=False)
        train_dataset.transform = transform
        train_loader = get_previous_train_loader(train_dataset, batch_size, self)

        return train_loader

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), SequentialMNIST.TRANSFORM])
        return transform

    @staticmethod
    def get_backbone():
        return resnet18(SequentialMNIST.N_CLASSES_PER_TASK
                        * SequentialMNIST.N_TASKS)

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize((0.1307,), (0.3081,))
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize((0.1307,), (0.3081,))
        return transform

