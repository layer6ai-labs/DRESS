#!/usr/bin/env python3
import sys
import random
from torch.utils.data import Dataset, Subset, ConcatDataset
from torchvision import transforms
from torchvision.datasets.omniglot import Omniglot

sys.path.append("../")
from partition_generators import generate_label_based_partition
from utils import *


def load_omniglot(args):
    """
    Benchmark definition for Omniglot.
    """
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        #lambda x: 1.0 - x, # originally was in learn2learn example for omniglot, not sure why
    ])
   # Set up both the background and eval dataset
    omni_background = Omniglot(DATADIR, 
                               background=True, 
                               download=True,
                               transform=data_transforms)
    # Eval labels also start from 0.
    # It's important to add 964 to label values in eval so they don't overwrite background dataset.
    omni_evaluation = Omniglot(DATADIR,
                               background=False,
                               download=True,
                               transform=data_transforms,
                               target_transform=lambda x: x + len(omni_background._characters))
    omni = ConcatDataset((omni_background, omni_evaluation))

    # split into meta_train, meta_valid, and meta_test datasets
    omni_clses = list(range(1623)) # total of 1623 characters
    random.shuffle(omni_clses)
    meta_train_clses, meta_valid_clses, meta_test_clses = omni_clses[:1100], omni_clses[1100:1200], omni_clses[1200:]
    meta_train_idxs, meta_valid_idxs, meta_test_idxs = [], [], []
    for i, (_, cls) in enumerate(omni):
        if cls in meta_train_clses:
            meta_train_idxs.append(i)
        elif cls in meta_valid_clses:
            meta_valid_idxs.append(i)
        elif cls in meta_test_clses:
            meta_test_idxs.append(i)
        else:
            print(f"Invalid class: {cls}!")
            exit(1)
    omni_meta_train, omni_meta_valid, omni_meta_test = Subset(omni, meta_train_idxs), Subset(omni, meta_valid_idxs), Subset(omni, meta_test_idxs)

    # simply use omniglot labels for generating tasks
    meta_train_partitions_supervised, meta_valid_partitions, meta_test_partitions = generate_label_based_partition(omni_meta_train), \
                                                                    generate_label_based_partition(omni_meta_valid), \
                                                                    generate_label_based_partition(omni_meta_test)

    # there is no common labels among meta splits, so for SupAll and SupOra there 
    # is no corresponding labels and partitions
    return (
        omni_meta_train, 
        omni_meta_valid, 
        omni_meta_test,  
        meta_train_partitions_supervised,  
        None,
        None,
        meta_valid_partitions,  
        meta_test_partitions  
    )