import random

import numpy as np

from PIL import ImageFilter

import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as F

class RandomNoise(object):
    def __init__(self, ratio):
        self.ratio = ratio

    def __call__(self, x):
        noise = np.random.choice([-1, 0, 1], x.shape[0], p=[self.ratio/2, 1-self.ratio, self.ratio/2])
        x = np.abs(x-noise)
        return x


class GaussianBlur(object):
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class MultipleTransform(nn.Module):
    def __init__(self, transforms):
        super().__init__()
        self.transforms = transforms

    def forward(self, x):
        return [t(x) for t in self.transforms]


# class FewShotTaskSampler(torch.utils.data.BatchSampler):
#     def __init__(self, dataset, N, K, Q, num_tasks):
#         self.N = N
#         self.K = K
#         self.Q = Q
#         self.num_tasks = num_tasks

#         if isinstance(dataset, (D.CIFAR10, D.CIFAR100, ISIC2018, ChestX)):
#             labels = dataset.targets
#         elif isinstance(dataset, (D.ImageFolder, MPI3DToy, Omniglot, JSONImageDataset, Cars)):
#             labels = [y for _, y in dataset.samples]
#         else:
#             raise NotImplementedError

#         self.indices = defaultdict(list)
#         for i, y in enumerate(labels):
#             self.indices[y].append(i)

#     def __iter__(self):
#         for _ in range(self.num_tasks):
#             batch_indices = []
#             labels = random.sample(list(self.indices.keys()), self.N)
#             for y in labels:
#                 if len(self.indices[y]) >= self.K+self.Q:
#                     batch_indices.extend(random.sample(self.indices[y], self.K+self.Q))
#                 else:
#                     batch_indices.extend(random.choices(self.indices[y], k=self.K+self.Q))
#             yield batch_indices

class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]
    
class CropCelebA(object):
    def __call__(self, img):
        new_img = F.crop(img, 57, 35, 128, 100)
        return new_img

def get_augmentation(dataset, method='none'):
    interpolation=T.InterpolationMode.BICUBIC
    if dataset == 'cifar10':
        mean = [0.4914, 0.4822, 0.4465]
        std  = [0.2023, 0.1994, 0.2010]
        if method == 'none':
            return T.Compose([T.ToTensor(),
                              T.Normalize(mean=mean, std=std)])
        elif method == 'strong':
            return T.Compose([T.RandomResizedCrop(32, scale=(0.2, 1.), interpolation=interpolation),
                              T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                              T.RandomGrayscale(p=0.2),
                              T.RandomHorizontalFlip(),
                              T.ToTensor(),
                              T.Normalize(mean=mean, std=std)])
        elif method == 'weak':
            return T.Compose([T.RandomResizedCrop(32, scale=(0.2, 1.), interpolation=interpolation),
                              T.RandomHorizontalFlip(),
                              T.ToTensor(),
                              T.Normalize(mean=mean, std=std)])

    elif dataset == 'miniimagenet':
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]
        if method == 'none':
            return T.Compose([T.ToTensor(),
                              T.Normalize(mean=mean, std=std)])
        elif method == 'strong':
            return T.Compose([T.RandomResizedCrop(84, scale=(0.2, 1.), interpolation=interpolation),
                              T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                              T.RandomGrayscale(p=0.2),
                              T.RandomHorizontalFlip(),
                              T.ToTensor(),
                              T.Normalize(mean=mean, std=std)])
        elif method == 'weak':
            return T.Compose([T.RandomResizedCrop(84, scale=(0.2, 1.), interpolation=interpolation),
                              T.RandomHorizontalFlip(),
                              T.ToTensor(),
                              T.Normalize(mean=mean, std=std)])

    elif dataset == 'omniglot':
        mean = [0.92206]
        std  = [0.08426]
        if method == 'none':
            return T.Compose([T.Resize((28, 28)),
                              T.ToTensor(),
                              T.Normalize(mean=mean, std=std)])
        elif method in ['strong', 'weak']:
            return T.Compose([T.RandomResizedCrop(28, scale=(0.2, 1.), interpolation=interpolation),
                              T.RandomHorizontalFlip(),
                              T.ToTensor(),
                              T.Normalize(mean=mean, std=std)])
        
    elif dataset.startswith("mpi3d"):
        mean = [0.2244, 0.2214, 0.2263]
        std = [0.0690, 0.0634, 0.0761]

        if method == 'none':
            return T.Compose([T.ToTensor(),
                              T.Normalize(mean=mean, std=std)])
        elif method in ['strong', 'weak']:
            return T.Compose([T.ToPILImage(),
                              T.RandomResizedCrop(84, scale=(0.4, 1.), interpolation=T.InterpolationMode.BICUBIC),
                              #T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                              #T.RandomGrayscale(p=0.2),
                              T.RandomHorizontalFlip(),
                              T.ToTensor(),
                              T.Normalize(mean=mean, std=std)])
    elif dataset == 'shapes3d':
        mean = [0.5036, 0.5788, 0.6034]
        std = [0.3493, 0.4011,  0.4212]

        if method == 'none':
            return T.Compose([T.ToTensor(),
                              T.Normalize(mean=mean, std=std)])
        elif method in ['strong', 'weak']:
            return T.Compose([T.ToPILImage(),
                              T.RandomResizedCrop(64, scale=(0.4, 1.), interpolation=T.InterpolationMode.BICUBIC),
                              #T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                              #T.RandomGrayscale(p=0.2),
                              T.RandomHorizontalFlip(),
                              T.ToTensor(),
                              T.Normalize(mean=mean, std=std)])
    elif dataset == 'norb':
        mean = [0.7521, 0.7521, 0.7521]
        std  = [0.1773, 0.1773, 0.1773]

        if method == 'none':
            return T.Compose([T.ToTensor(),
                              T.Lambda(lambda x: x.repeat(3, 1, 1)),
                              T.Normalize(mean=mean, std=std)])
        elif method in ['strong', 'weak']:
            return T.Compose([T.ToPILImage(),
                              T.RandomResizedCrop(64, scale=(0.4, 1.), interpolation=T.InterpolationMode.BICUBIC),
                              #T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                              #T.RandomGrayscale(p=0.2),
                              T.RandomHorizontalFlip(),
                              T.ToTensor(),
                              T.Lambda(lambda x: x.repeat(3, 1, 1)),
                              T.Normalize(mean=mean, std=std)])
    elif dataset == 'causal3d':
        mean = [0.4327, 0.2689, 0.2839]
        std = [0.1164, 0.0795, 0.0840]
        
        if method == 'none':
            return T.Compose([T.ToTensor(),
                              T.Normalize(mean=mean, std=std)])
        elif method in ['strong', 'weak']:
            return T.Compose([T.RandomResizedCrop(64, scale=(0.4, 1.), interpolation=T.InterpolationMode.BICUBIC),
                              #T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                              #T.RandomGrayscale(p=0.2),
                              T.RandomHorizontalFlip(),
                              T.ToTensor(),
                              T.Normalize(mean=mean, std=std)])
            
    elif dataset.startswith('celeba'):
        mean = [0.5773, 0.4355, 0.3658]
        std  = [0.2459, 0.2096, 0.1954]
        if method == 'none':
            return T.Compose([CropCelebA(),
                              T.Resize(size=(128, 128)),
                              T.ToTensor(),
                              T.Normalize(mean=mean, std=std)])
        elif method == 'strong':
            return T.Compose([CropCelebA(),
                              T.RandomResizedCrop(128, scale=(0.2, 1.), interpolation=interpolation),
                              T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                              T.RandomGrayscale(p=0.2),
                              T.RandomApply([GaussianBlur()], p=0.5),
                              T.RandomHorizontalFlip(),
                              T.ToTensor(),
                              T.Normalize(mean=mean, std=std)])
        elif method == 'weak':
            return T.Compose([CropCelebA(),
                              T.RandomResizedCrop(128, scale=(0.2, 1.), interpolation=interpolation),
                              T.RandomHorizontalFlip(),
                              T.ToTensor(),
                              T.Normalize(mean=mean, std=std)])

    elif dataset in ['imagenet']:
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]
        if method == 'none':
            return T.Compose([T.Resize(256),
                              T.CenterCrop(224),
                              T.ToTensor(),
                              T.Normalize(mean=mean, std=std)])
        elif method == 'strong':
            return T.Compose([T.RandomResizedCrop(224, scale=(0.2, 1.), interpolation=interpolation),
                              T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                              T.RandomGrayscale(p=0.2),
                              T.RandomApply([GaussianBlur()], p=0.5),
                              T.RandomHorizontalFlip(),
                              T.ToTensor(),
                              T.Normalize(mean=mean, std=std)])
        elif method == 'weak':
            return T.Compose([T.RandomResizedCrop(224, scale=(0.2, 1.), interpolation=interpolation),
                              T.RandomHorizontalFlip(),
                              T.ToTensor(),
                              T.Normalize(mean=mean, std=std)])

    elif isinstance(dataset, list) and dataset[1] in ['cub200', 'cropdiseases', 'eurosat', 'isic', 'chestx', 'places', 'cars', 'plantae']:
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]
        if dataset[0] == 'imagenet':
            return T.Compose([T.Resize(256, interpolation=interpolation),
                              T.CenterCrop(224),
                              T.ToTensor(),
                              T.Normalize(mean=mean, std=std)])

        else:
            return T.Compose([T.Resize(84, interpolation=interpolation),
                              T.CenterCrop(84),
                              T.ToTensor(),
                              T.Normalize(mean=mean, std=std)])

    else:
        raise NotImplementedError