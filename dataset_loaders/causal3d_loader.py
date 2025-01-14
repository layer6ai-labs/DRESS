#!/usr/bin/env python3
import sys
import os
import random
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset
from torchvision import transforms
from torchvision.datasets.celeba import CelebA
from torchvision.utils import save_image

sys.path.append("../")
from partition_generators import generate_attributes_based_partitions
from utils import *

CAUSAL3D_DIR = {
    "train": os.path.join(DATADIR, "causal3d", "train"),
    "test": os.path.join(DATADIR, "causal3d", "test")
}
N_OBJ_CLS = 7
N_IMGS_PER_SUBDIR = 36_000
N_ATTRS = 10

class Causal3D(Dataset):
    def __init__(self, imgs, attrs, transforms):
        assert len(imgs) == len(attrs)
        self.imgs = imgs
        self.attrs = np.array(attrs)
        assert np.shape(self.attrs) == (len(imgs), N_ATTRS)
        self.transforms = transforms

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_tensor, attrs_tensor = self.transforms(self.imgs[index]), torch.tensor(self.attrs[index])
        return (img_tensor, attrs_tensor)

def discretize_causal3d_attrs(attrs_raw):
    return 

def _load_causal3d(args):
    # Resize happens later in the pipeline
    img_transforms = transforms.Compose([
        transforms.ToTensor()
    ])

    ds_train, ds_test = None, None
    for split in ["train", "test"]:
        imgs_all, attrs_raw_all = [], []
        for obj_cls in range(N_OBJ_CLS):
            img_dir = os.path.join(CAUSAL3D_DIR[split], f"images_{obj_cls}")
            attrs_raw_onecls = np.load(CAUSAL3D_DIR[split], f"latents_{obj_cls}.npy")
            assert np.shape(attrs_raw_onecls) == (N_IMGS_PER_SUBDIR, N_ATTRS)
            for i in range(N_IMGS_PER_SUBDIR):
                img_filename = f"{i:05d}.png"
                imgs_all.append(Image.open(os.path.join(img_dir, img_filename)))
                attrs_raw_all.append(attrs_raw_onecls[i])
        attrs_all = discretize_causal3d_attrs(attrs_raw_all)
        if split=="train":
            ds_train = Causal3D(imgs_all, attrs_all, img_transforms)
        else:
            ds_test = Causal3D(imgs_all, attrs_all, img_transforms)

    # just a placeholder
    ds_valid = ds_train

    CAUSAL3D_ATTRIBUTES_COUNTS = [5,5,18,9,6]
    CAUSAL3D_ATTRIBUTES_IDX_META_TRAIN = [0,1] # about object identity
    CAUSAL3D_ATTRIBUTES_IDX_META_VALID = [0,1] # without early stopping, meta validation doesn't matter
    CAUSAL3D_ATTRIBUTES_IDX_META_TEST = [2,3,4] # pose and lighting condition

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


def load_causal3d(args):
    return _load_causal3d(args)