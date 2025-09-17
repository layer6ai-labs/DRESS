#!/usr/bin/env python3
import os
import sys
import numpy as np
from torch.utils.data import Dataset 
from torchvision import transforms as T
import h5py

sys.path.append("../")
from partition_generators import generate_attributes_based_partitions
from utils import *


class Shapes3D(Dataset):
    def __init__(self, imgs, attrs, transforms):
        self.imgs = imgs
        self.attrs = attrs
        self.transform = transforms

    def __len__(self):
        return np.shape(self.imgs)[0]

    def __getitem__(self, index):
        return (self.transform(self.imgs[index]), torch.tensor(self.attrs[index]))
    

def load_shapes3d(args):
    datafile_path = os.path.join(DATADIR, "shapes3d", "shapes3d.h5")
    print(f"Loading shapes3d data from {datafile_path}...")
    shapes3d_data = h5py.File(datafile_path, 'r')
    shapes3d_imgs = shapes3d_data['images'][()]
    n_imgs = shapes3d_imgs.shape[0]
    # The data is indexed with the following dimension arrangement, 
    # corresponding to the six factors:
    # 10 X 10 X 10 X 8 X 4 X 15
    shapes3d_attrs = shapes3d_data['labels'][()]
    assert np.shape(shapes3d_imgs) == (n_imgs, 64, 64, 3) and \
            np.shape(shapes3d_attrs) == (n_imgs, 6)
    
    assert n_imgs == 480_000
    n_train_imgs = 400_000
    n_test_imgs = 50_000

    # convert attributes to integers (starting from 0) for generating partitions
    attrs_counts = [10, 10, 10, 8, 4, 15]
    attrs_offsets = [0,  0,  0, -0.75, None, 30]
    attrs_stepsizes = [10, 10, 10, 14, None, 14/60]
    for i, (offset, stepsize) in enumerate(zip(attrs_offsets, attrs_stepsizes)):
        if offset is None:
            # this attribute already in the nature number format
            continue
        shapes3d_attrs[:, i] = (shapes3d_attrs[:, i] + offset) * stepsize  
    
    print(f"[Shapes3D] Getting meta splits.....")
    # get meta split indices
    perm = np.arange(n_imgs)
    np.random.shuffle(perm)
    metatrain_idxs, metatest_idxs = \
                perm[:n_train_imgs], perm[-n_test_imgs:]
    
    data_transforms = build_initial_img_transforms(meta_split="meta_train", args=args)
    metatrain_dataset = Shapes3D(shapes3d_imgs[metatrain_idxs], shapes3d_attrs[metatrain_idxs], data_transforms)
    data_transforms = build_initial_img_transforms(meta_split="meta_test", args=args)
    metatest_dataset = Shapes3D(shapes3d_imgs[metatest_idxs], shapes3d_attrs[metatest_idxs], data_transforms)
    
    metatrain_attrs_all, metatest_attrs = \
                shapes3d_attrs[metatrain_idxs], shapes3d_attrs[metatest_idxs]
    
    """
    The attributes with order: floor color, wall color, object color, scale, shape, orientation
    """
    SHAPES3D_ATTRIBUTES_IDX_META_TRAIN = [2,3,4]
    SHAPES3D_ATTRIBUTES_IDX_META_TEST = [0,1,5]

    metatrain_attrs = metatrain_attrs_all[:, SHAPES3D_ATTRIBUTES_IDX_META_TRAIN]
    metatest_attrs = metatest_attrs[:, SHAPES3D_ATTRIBUTES_IDX_META_TEST]

    metatrain_attrs_oracle = metatrain_attrs_all[:, SHAPES3D_ATTRIBUTES_IDX_META_TEST]

    # generate partitions with binary classification on celeba attributes
    metatrain_partitions_supervised = generate_attributes_based_partitions(
                                        metatrain_attrs, 
                                        np.array(attrs_counts)[SHAPES3D_ATTRIBUTES_IDX_META_TRAIN], 
                                        'meta_train', 
                                        args)

    metatest_partitions = generate_attributes_based_partitions(
                                        metatest_attrs, 
                                        np.array(attrs_counts)[SHAPES3D_ATTRIBUTES_IDX_META_TEST],
                                        'meta_test', 
                                        args)
    
    metatrain_partitions_supervised_all = generate_attributes_based_partitions(
                                        metatrain_attrs_all,
                                        np.array(attrs_counts),
                                        'meta_train',
                                        args)

    metatrain_partitions_supervised_oracle = generate_attributes_based_partitions(
                                        metatrain_attrs_oracle,
                                        np.array(attrs_counts)[SHAPES3D_ATTRIBUTES_IDX_META_TEST],
                                        'meta_train',
                                        args)

    return (
        metatrain_dataset, 
        metatest_dataset,  
        metatrain_partitions_supervised,  
        metatrain_partitions_supervised_all,
        metatrain_partitions_supervised_oracle,
        metatest_partitions  
    )

    
