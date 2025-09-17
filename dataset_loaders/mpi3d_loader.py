#!/usr/bin/env python3
import os
import sys
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms as T

sys.path.append("../")
from partition_generators import generate_attributes_based_partitions
from utils import *


MPI3D_ATTRIBUTES_NUM_ANGULAR_VALUES = 10

class MPI3D(Dataset):
    def __init__(self, imgs, attrs, transforms):
        self.imgs = imgs
        self.attrs = attrs
        self.transform = transforms

    def __len__(self):
        return np.shape(self.imgs)[0]

    def __getitem__(self, index):
        return (self.transform(self.imgs[index]), torch.tensor(self.attrs[index]))



def _load_mpi3d(args, meta_split_type):
    datafile_path = os.path.join(DATADIR, "mpi3d", "mpi3d_toy.npz")
    print(f"Loading mpi3d data from {datafile_path}...")
    mpi3d_imgs = np.load(datafile_path)['images']
    n_imgs = mpi3d_imgs.shape[0]
    
    assert n_imgs == 1_036_800
    n_train_imgs = 1_000_000
    n_test_imgs = 30_000
    # The data is indexed with the following dimension arrangement, 
    # corresponding to the seven factors:
    # 6 X 6 X 2 X 3 X 3 X 40 X 40
    MPI3D_ATTRIBUTES_COUNTS = [6,6,2,3,3,40,40]

    # get the seven factor values of each image by ordered indices
    idx_bases = np.prod(MPI3D_ATTRIBUTES_COUNTS)/np.cumprod(MPI3D_ATTRIBUTES_COUNTS)
    mpi3d_attrs = []
    remainders = np.arange(n_imgs)
    for i in range(len(MPI3D_ATTRIBUTES_COUNTS)):
        mpi3d_attrs_one_dim, remainders = np.divmod(remainders, idx_bases[i])
        mpi3d_attrs.append(np.expand_dims(mpi3d_attrs_one_dim, axis=1))
    mpi3d_attrs = np.concatenate(mpi3d_attrs,axis=1)
    assert np.shape(mpi3d_attrs) == (n_imgs, 7)

    # process the last two attributes, robot arm horizontal axis, robot arm vertical axis
    # originally each of them has 40 values (4.5 degrees interval, too hard to identify even for human, under the camera height variation)
    if MPI3D_ATTRIBUTES_NUM_ANGULAR_VALUES != 40:
        assert MPI3D_ATTRIBUTES_NUM_ANGULAR_VALUES < 40
        mpi3d_attrs[:, -1] *= (MPI3D_ATTRIBUTES_NUM_ANGULAR_VALUES / 40) 
        mpi3d_attrs[:, -2] *= (MPI3D_ATTRIBUTES_NUM_ANGULAR_VALUES / 40)
        mpi3d_attrs[:, -1], mpi3d_attrs[:, -2] = \
            np.floor(mpi3d_attrs[:,-1]), np.floor(mpi3d_attrs[:,-2])
        
    MPI3D_ATTRIBUTES_COUNTS[-1], MPI3D_ATTRIBUTES_COUNTS[-2] = \
        MPI3D_ATTRIBUTES_NUM_ANGULAR_VALUES, MPI3D_ATTRIBUTES_NUM_ANGULAR_VALUES
    
    
    print(f"[MPI3D_{meta_split_type}] Getting meta splits.....")
    # get meta split indices
    perm = np.arange(n_imgs)
    np.random.shuffle(perm)
    metatrain_idxs, metatest_idxs = \
                perm[:n_train_imgs], perm[-n_test_imgs:]
    
    data_transforms = build_initial_img_transforms(meta_split="meta_train", args=args)
    metatrain_dataset = MPI3D(mpi3d_imgs[metatrain_idxs], mpi3d_attrs[metatrain_idxs], data_transforms)
    data_transforms = build_initial_img_transforms(meta_split="meta_test", args=args)
    metatest_dataset = MPI3D(mpi3d_imgs[metatest_idxs], mpi3d_attrs[metatest_idxs], data_transforms)
    
    metatrain_attrs_all, metatest_attrs = \
                mpi3d_attrs[metatrain_idxs], mpi3d_attrs[metatest_idxs]
    
    """
    The attributes with order: object color, object shape, object size, camera height, background color, horizontal axis, vertical axis
    """
    if meta_split_type == "hard":
        MPI3D_ATTRIBUTES_IDX_META_TRAIN = [0,1,2]
        MPI3D_ATTRIBUTES_IDX_META_TEST = [5,6]
    else:
        MPI3D_ATTRIBUTES_IDX_META_TRAIN = [0,1,2]
        MPI3D_ATTRIBUTES_IDX_META_TEST = [3,4]

    metatrain_attrs = metatrain_attrs_all[:, MPI3D_ATTRIBUTES_IDX_META_TRAIN]
    metatest_attrs = metatest_attrs[:, MPI3D_ATTRIBUTES_IDX_META_TEST]

    metatrain_attrs_oracle = metatrain_attrs_all[:, MPI3D_ATTRIBUTES_IDX_META_TEST]

    # generate partitions with binary classification on celeba attributes
    metatrain_partitions_supervised = generate_attributes_based_partitions(
                                        metatrain_attrs, 
                                        np.array(MPI3D_ATTRIBUTES_COUNTS)[MPI3D_ATTRIBUTES_IDX_META_TRAIN], 
                                        'meta_train', 
                                        args)
    
    metatest_partitions = generate_attributes_based_partitions(
                                        metatest_attrs, 
                                        np.array(MPI3D_ATTRIBUTES_COUNTS)[MPI3D_ATTRIBUTES_IDX_META_TEST],
                                        'meta_test', 
                                        args)
    
    metatrain_partitions_supervised_all = generate_attributes_based_partitions(
                                        metatrain_attrs_all,
                                        np.array(MPI3D_ATTRIBUTES_COUNTS),
                                        'meta_train',
                                        args)

    metatrain_partitions_supervised_oracle = generate_attributes_based_partitions(
                                        metatrain_attrs_oracle,
                                        np.array(MPI3D_ATTRIBUTES_COUNTS)[MPI3D_ATTRIBUTES_IDX_META_TEST],
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

def load_mpi3d_easy(args):
    return _load_mpi3d(args, meta_split_type='easy')

def load_mpi3d_hard(args):
    return _load_mpi3d(args, meta_split_type='hard')