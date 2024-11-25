#!/usr/bin/env python3
import os
import sys
import numpy as np
from torch.utils.data import Dataset, Subset 
from torchvision import transforms as T
from torchvision.datasets import ImageFolder

sys.path.append("../")
from partition_generators import generate_attributes_based_partitions
from utils import *


class Animals(Dataset):
    def __init__(self, imgs_dir, attrs_by_cls_filename, data_transforms):
        # construct the dataset from folders
        self.ds_orig = ImageFolder(root=imgs_dir, transform=data_transforms)
        # load attributes for all classes
        attr_lines = open(attrs_by_cls_filename, "r").readlines()
        attrs_cls_all = []
        for attr_line in attr_lines:
            attrs_cls_all.append(np.array(attr_line.strip().split(), dtype=int))
        attrs_cls_all = np.stack(attrs_cls_all, axis=0)
        assert np.shape(attrs_cls_all) == (50, 85)
        # organize attributes sample by sample following same indexing
        print("[Animals] Preparing attributes")
        attrs_all = [attrs_cls_all[cls] for _, cls in self.ds_orig]
        self.attrs_all = np.stack(attrs_all, axis=0)
        assert np.shape(self.attrs_all) == (len(self.ds_orig), 85)

    def __len__(self):
        return len(self.ds_orig)

    def __getitem__(self, index):
        img, attrs = self.ds_orig[index], self.attrs_all[index]
        return (img, torch.tensor(attrs))


def load_animals(args):
    # Resize happens later in the pipeline
    data_transforms = T.Compose([
        T.ToTensor()
    ])
    datadir_path = os.path.join(DATADIR, "animals", "JPEGImages")
    attr_by_cls_filename = os.path.join(DATADIR, "animals", "predicate-matrix-binary.txt")
    print(f"Loading animals data from {datadir_path}...")
    ds = Animals(datadir_path, attr_by_cls_filename, data_transforms)
    
    n_imgs = len(ds)
    assert n_imgs == 37_322
    n_train_imgs = 25_000
    n_val_imgs = 322
    n_test_imgs = 12_000
    assert n_imgs == n_train_imgs + n_val_imgs + n_test_imgs
    
    print(f"[Animals] Getting meta splits.....")
    # get meta split indices
    perm = np.arange(n_imgs)
    np.random.shuffle(perm)
    metatrain_idxs, metavalid_idxs, metatest_idxs = \
                perm[:n_train_imgs], perm[n_train_imgs:n_train_imgs+n_val_imgs], perm[-n_test_imgs:]
    
    # meta training (or pre-training)
    (
        metatrain_dataset, 
        metavalid_dataset, 
        metatest_dataset
    ) = Subset(ds, metatrain_idxs), Subset(ds, metavalid_idxs), Subset(ds, metatest_idxs)
    
    metatrain_attrs_all, metavalid_attrs, metatest_attrs = \
                ds.attrs_all[metatrain_idxs], ds.attrs_all[metavalid_idxs], ds.attrs_all[metatest_idxs]
    
    
    ANIMALS_ATTRIBUTES_IDX_META_TRAIN = np.arange(55)
    ANIMALS_ATTRIBUTES_IDX_META_VALID = np.arange(5) # don't matter without early stopping
    ANIMALS_ATTRIBUTES_IDX_META_TEST = np.arange(55,85)

    metatrain_attrs = metatrain_attrs_all[:, ANIMALS_ATTRIBUTES_IDX_META_TRAIN]
    metavalid_attrs = metavalid_attrs[:, ANIMALS_ATTRIBUTES_IDX_META_VALID]
    metatest_attrs = metatest_attrs[:, ANIMALS_ATTRIBUTES_IDX_META_TEST]

    metatrain_attrs_oracle = metatrain_attrs_all[:, ANIMALS_ATTRIBUTES_IDX_META_TEST]

    # generate partitions with binary classification on celeba attributes
    metatrain_partitions_supervised = generate_attributes_based_partitions(
                                        metatrain_attrs, 
                                        2, 
                                        'meta_train', 
                                        args)
    metavalid_partitions = generate_attributes_based_partitions(
                                        metavalid_attrs, 
                                        2, 
                                        'meta_valid', 
                                        args)
    metatest_partitions = generate_attributes_based_partitions(
                                        metatest_attrs, 
                                        2,
                                        'meta_test', 
                                        args)
    
    metatrain_partitions_supervised_all = generate_attributes_based_partitions(
                                        metatrain_attrs_all,
                                        2,
                                        'meta_train',
                                        args)

    metatrain_partitions_supervised_oracle = generate_attributes_based_partitions(
                                        metatrain_attrs_oracle,
                                        2,
                                        'meta_train',
                                        args)

    return (
        metatrain_dataset, 
        metavalid_dataset, 
        metatest_dataset,  
        metatrain_partitions_supervised,  
        metatrain_partitions_supervised_all,
        metatrain_partitions_supervised_oracle,
        metavalid_partitions,  
        metatest_partitions  
    )


