import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from backbone.ResNet18 import resnet18
import torch.nn.functional as F
from utils.conf import base_path
from PIL import Image
import numpy as np
import torch
from .utils.continual_dataset import ContinualDataset, store_loaders
from .utils.continual_dataset import get_previous_train_loader
from typing import Tuple
from datasets.transforms.denormalization import DeNormalize
from skimage import io, transform
from tqdm.notebook import tqdm


#for CINIC-10
def get_train_val_cinic(train, test_transform: transforms, dataset: str, val_perc: float=0.1):
    """
    Extract val_perc% of the training set as the validation set.
    :param train: training dataset
    :param test_transform: transform for the validation set
    :param dataset: name of the dataset
    :param val_perc: percentage of validation set
    :return: the training set and the validation set
    """
    dataset_length = len(train)
    val_size = int(val_perc * dataset_length)
    train_size = dataset_length - val_size
    seed = 42
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = torch.utils.data.random_split(train, [train_size, val_size],generator=generator)

    # Apply the transformation to the validation set
    val_dataset.dataset.transform = test_transform

    return train_dataset, val_dataset

class MyCINIC10(Dataset):
    """
    Custom dataset class for CINIC-10.
    """
    def __init__(self, root,transform=None, target_transform=None):
        super().__init__()
#         print("Loader: {}".format(loader))
#         assert loader is not None, "Loader is found None"
        self.transform = transform
        self.target_transform = target_transform
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
       
        self.class_names = os.listdir(root)
        self.samples = []
        skipped_imgs = 0
        for class_label, class_name in enumerate(self.class_names):
            files = os.listdir(os.path.join(root, class_name))
#             print(os.path.join(root, class_name))
            for img in tqdm(files, desc=f"Loading Class: {class_name}"):
                img_path = os.path.join(root, class_name, img)
                img = io.imread(img_path)
                
                if len(img.shape) < 3:
#                     print(f"Skipping {img_path} because the shape is {img.shape}")
                    skipped_imgs += 1
                else:
                    self.samples.append((img_path, class_label))
        print(f"Total IMGs remaining: {len(self.samples)}, Skipped: {skipped_imgs}")   
    
    def __getitem__(self, index: int) -> Tuple[type(Image), int]:
        """
        Gets the requested element from the dataset.
        :param index: index of the element to be returned
        :returns: tuple: (image, target) where target is index of the target class.
        """
        path, target = self.samples[index]
        img = io.imread(path)
#         print(f"Image shape: {img.shape}, path: {path}")
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
    def __len__(self):
        return len(self.samples)


class CINIC10(ContinualDataset):

    NAME = 'CINIC10'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 1
    N_TASKS = 10
    TRANSFORM = transforms.Compose(
            [transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize((0.47889522, 0.47227842, 0.43047404),
                                  (0.24205776, 0.23828046, 0.25874835))])

    def get_data_loaders(self):
        transform = self.TRANSFORM

        test_transform = transforms.Compose(
            [transforms.ToTensor(), self.get_normalization_transform()])
        
        dataset_path_train = os.path.join(os.getcwd(), 'datasets', 'CINIC-10/train')
        dataset_path_test = os.path.join(os.getcwd(), 'datasets', 'CINIC-10/test')
        train_dataset = MyCINIC10(dataset_path_train, transform=transform)
        if self.args.validation:
            train_dataset, test_dataset = get_train_val_cinic(train_dataset,
                                                    test_transform, self.NAME)
        else:
            test_dataset = MyCINIC10(dataset_path_test, transform=test_transform)

        train, test = store_loaders(train_dataset, test_dataset, self)
        return train, test

    def not_aug_dataloader(self, batch_size):
        transform = transforms.Compose([transforms.ToTensor(), self.get_normalization_transform()])

        train_dataset = MyCINIC10(dataset_path_train, transform=transform)
        train_loader = get_previous_train_loader(train_dataset, batch_size, self)

        return train_loader

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), CINIC10.TRANSFORM])
        return transform

    @staticmethod
    def get_backbone():
        return resnet18(CINIC10.N_CLASSES_PER_TASK
                        * CINIC10.N_TASKS)

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize((0.47889522, 0.47227842, 0.43047404),
                                         (0.24205776, 0.23828046, 0.25874835))
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize((0.47889522, 0.47227842, 0.43047404),
                                (0.24205776, 0.23828046, 0.25874835))
        return transform

