import torch
from torchvision import datasets

from dataset import get_semi_supervised_dataset
from easyfl.datasets.data import CIFAR100, CIFAR10, MiniImageNetDataset
from transform import SimCLRTransform
import numpy as np
import random
import torchvision
import copy
from torchvision.transforms import ToPILImage, RandomCrop, RandomHorizontalFlip, ToTensor, Normalize
def get_data_loaders(dataset, image_size=32, batch_size=512, num_workers=8):
    transformation = SimCLRTransform(size=image_size, gaussian=False).test_transform

    if dataset == CIFAR100:
        data_path = "./data/cifar100"
        train_dataset = datasets.CIFAR100(data_path, download=True, transform=transformation)
        test_dataset = datasets.CIFAR100(data_path, train=False, download=True, transform=transformation)
    elif dataset == CIFAR10:
        data_path = "./data/cifar10"
        train_dataset = datasets.CIFAR10(data_path, download=True, transform=transformation)
        test_dataset = datasets.CIFAR10(data_path, train=False, download=True, transform=transformation)
    elif dataset == 'mini_imagenet':

        transform_train = torchvision.transforms.Compose([
            RandomHorizontalFlip(p=0.5),
            ToTensor(),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        transform_test = torchvision.transforms.Compose([
            ToTensor(),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        dataset = MiniImageNetDataset('./mini_imagenet/', transform_train)
        num_samples = len(dataset.image_paths)
        indices = np.arange(num_samples)
        train_indices = random.sample(list(indices), 50000)
        test_indices = list(set(indices) - set(train_indices))
        train_data = copy.deepcopy(dataset)
        train_data.image_paths = [train_data.image_paths[i] for i in train_indices]
        train_data.labels = [train_data.labels[i] for i in train_indices]
        test_data = MiniImageNetDataset('./mini_imagenet/', transform_test)
        test_data.image_paths = [test_data.image_paths[i] for i in test_indices]
        test_data.labels = [test_data.labels[i] for i in test_indices]
        train_dataset = train_data
        test_dataset = test_data

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

    return train_loader, test_loader


def get_semi_supervised_data_loaders(dataset, data_distribution, class_per_client, label_ratio, batch_size=512, num_workers=8, image_size=32):
    transformation = SimCLRTransform(size=image_size, gaussian=False).test_transform
    if dataset == CIFAR100:
        data_path = "./data/cifar100"
        test_dataset = datasets.CIFAR100(data_path, train=False, download=True, transform=transformation)
    else:
        data_path = "./data/cifar10"
        test_dataset = datasets.CIFAR10(data_path, train=False, download=True, transform=transformation)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

    _, _, labeled_data = get_semi_supervised_dataset(dataset, 5, data_distribution, class_per_client, label_ratio)
    return labeled_data.loader(batch_size), test_loader
