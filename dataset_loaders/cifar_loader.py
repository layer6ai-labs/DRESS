import sys
sys.path.append("../")
from utils import *
from partition_generators import generate_supervised_partitions_single_attribute

from torchvision.datasets import CIFAR100
from torch.utils.data import Subset

METATRAIN_CLASSES = set(range(60))
METATEST_CLASSES = set(range(60, 100))

def filter_dataset_by_classes(ds, clses_to_include):
    indices = [i for i, label in enumerate(ds.targets) 
                    if label in clses_to_include]
    return Subset(ds, indices) 

def load_cifar(args):
    # meta-training set
    data_transforms = build_initial_img_transforms(meta_split="meta_train", args=args)
    meta_train_ds = CIFAR100(DATADIR,
                            train=True,   
                            download=True,
                            transform=data_transforms
                            )
    # meta-testing set
    data_transforms = build_initial_img_transforms(meta_split="meta_test", args=args)
    meta_test_ds = CIFAR100(DATADIR,
                            train=False,   
                            download=True,
                            transform=data_transforms
                            )
    
    # filter data such that meta-train and meta-test set are disjoint in classes
    meta_train_ds, meta_test_ds = filter_dataset_by_classes(meta_train_ds, METATRAIN_CLASSES), \
                                    filter_dataset_by_classes(meta_test_ds, METATEST_CLASSES)
    
    # use original class labels for generating tasks
    (
        meta_train_partitions_supervised, 
        meta_test_partitions
    ) = generate_supervised_partitions_single_attribute([label for img, label in meta_train_ds]), \
            generate_supervised_partitions_single_attribute([label for img, label in meta_test_ds])

    # there is no common labels among meta splits, so for SupAll and SupOra there 
    # is no corresponding labels and partitions
    return (
        meta_test_ds, 
        meta_test_ds,  
        meta_train_partitions_supervised,  
        None,
        None,
        meta_test_partitions  
    )

