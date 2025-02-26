#!/usr/bin/env python3
import sys
import os
import random
from tqdm import tqdm
import tensorflow_datasets as tfds
from torch.utils.data import Dataset
from torchvision import transforms as T

sys.path.append("../")
from partition_generators import generate_attributes_based_partitions
from utils import *

class Norb(Dataset):
    def __init__(self, ds_tf, img_transforms, instance_label_transform):
        self.images, self.attrs = [], []
        self.transform = img_transforms
        # build the images and attributes into tensors
        for sample in tqdm(ds_tf):
            (
                img, 
                category,
                inst, 
                azimuth, 
                elevation, 
                lighting
            ) = sample['image'], \
                sample['label_category'], \
                sample['instance'], \
                sample['label_azimuth'], \
                sample['label_elevation'], \
                sample['label_lighting']
            # the image here would be numpy array
            # with pixel value in [0,255]
            self.images.append(img) 
            attrs_one_img = np.array([category,
                                      instance_label_transform[inst], 
                                      azimuth, 
                                      elevation, 
                                      lighting])
            self.attrs.append(attrs_one_img)
        self.images = np.stack(self.images, axis=0)
        self.attrs = np.stack(self.attrs, axis=0)
        print("Norb dataset initialized!")

    def __len__(self):
        return np.shape(self.attrs)[0]

    def __getitem__(self, index):
        img, attrs = self.images[index], self.attrs[index]
        return (self.transform(img), torch.tensor(attrs))
    

def load_norb(args):
    ds_tf_train = tfds.as_numpy(
                    tfds.load('smallnorb', 
                              data_dir=DATADIR,
                              split='train', 
                              shuffle_files=False))
    instance_label_transform_train = {4:0, 6:1, 7:2, 8:3, 9:4}
    # for now, validation is just a place holder
    ds_tf_valid = tfds.as_numpy(
                    tfds.load('smallnorb', 
                              data_dir=DATADIR,
                              split='train', 
                              shuffle_files=False))
    instance_label_transform_valid = instance_label_transform_train
    ds_tf_test = tfds.as_numpy(
                    tfds.load('smallnorb',
                              data_dir=DATADIR,
                              split='test',
                              shuffle_files=False))
    instance_label_transform_test = {0:0, 1:1, 2:2, 3:3, 5:4}
    data_transforms = build_initial_img_transforms(meta_split="meta_train", args=args)
    ds_train, ds_valid = (Norb(ds_tf_train, data_transforms, instance_label_transform_train),
                          Norb(ds_tf_valid, data_transforms, instance_label_transform_valid))
    data_transforms = build_initial_img_transforms(meta_split="meta_test", args=args)
    ds_test = Norb(ds_tf_test, data_transforms, instance_label_transform_test)

    NORB_ATTRIBUTES_COUNTS = [5,5,18,9,6]
    NORB_ATTRIBUTES_IDX_META_TRAIN = [0,1] # about object identity
    NORB_ATTRIBUTES_IDX_META_VALID = [0,1] # without early stopping, meta validation doesn't matter
    NORB_ATTRIBUTES_IDX_META_TEST = [2,3,4] # pose and lighting condition

    # Use disjoint subset of attrs for meta splits
    norb_meta_train_attrs_all = ds_train.attrs
    norb_meta_train_attrs = ds_train.attrs[:,NORB_ATTRIBUTES_IDX_META_TRAIN]
    norb_meta_valid_attrs = ds_valid.attrs[:,NORB_ATTRIBUTES_IDX_META_VALID]
    norb_meta_test_attrs = ds_test.attrs[:,NORB_ATTRIBUTES_IDX_META_TEST]

    norb_meta_train_attrs_oracle = ds_train.attrs[:,NORB_ATTRIBUTES_IDX_META_TEST]

    # generate partitions with binary classification on celeba attrs
    meta_train_partitions_supervised = generate_attributes_based_partitions(
                                            norb_meta_train_attrs, 
                                            np.array(NORB_ATTRIBUTES_COUNTS)[NORB_ATTRIBUTES_IDX_META_TRAIN],
                                            'meta_train', 
                                            args)
    meta_valid_partitions = generate_attributes_based_partitions(
                                            norb_meta_valid_attrs, 
                                            np.array(NORB_ATTRIBUTES_COUNTS)[NORB_ATTRIBUTES_IDX_META_VALID],
                                            'meta_valid', 
                                            args)
    meta_test_partitions = generate_attributes_based_partitions(
                                            norb_meta_test_attrs, 
                                            np.array(NORB_ATTRIBUTES_COUNTS)[NORB_ATTRIBUTES_IDX_META_TEST],
                                            'meta_test', 
                                            args)

    meta_train_partitions_supervised_all = generate_attributes_based_partitions(
                                            norb_meta_train_attrs_all, 
                                            np.array(NORB_ATTRIBUTES_COUNTS),
                                            'meta_train', 
                                            args)
    
    meta_train_partitions_supervised_oracle = generate_attributes_based_partitions(
                                                norb_meta_train_attrs_oracle,
                                                np.array(NORB_ATTRIBUTES_COUNTS)[NORB_ATTRIBUTES_IDX_META_TEST],
                                                'meta_train',
                                                args)

    return (
        ds_train,  
        ds_valid,  
        ds_test,   
        meta_train_partitions_supervised, 
        meta_train_partitions_supervised_all,
        meta_train_partitions_supervised_oracle, 
        meta_valid_partitions,  
        meta_test_partitions
    )



    