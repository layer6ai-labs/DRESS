#!/usr/bin/env python3
import sys
import os
import random
from tqdm import trange
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

sys.path.append("../")
from partition_generators import generate_attributes_based_partitions
from utils import *

CAUSAL3D_DIR = {
    "train": os.path.join(DATADIR, "causal3d", "train"),
    "test": os.path.join(DATADIR, "causal3d", "test")
}
N_OBJ_CLS = 7
N_IMGS_PER_SUBDIR = {
    "train": 36_000,
    "test": 3_600
}
N_ATTRS = 10
N_LEVELS_PER_ATTR = 10


class Causal3D(Dataset):
    def __init__(self, imgs, attrs, transforms):
        assert len(imgs) == len(attrs)
        self.imgs = imgs
        self.attrs = attrs
        assert np.shape(self.attrs) == (len(imgs), N_ATTRS)
        self.transform = transforms

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_tensor, attrs_tensor = self.transform(self.imgs[index]), torch.tensor(self.attrs[index])
        return (img_tensor, attrs_tensor)

def discretize_causal3d_attrs(attrs_raw):
    attrs_quantized = []
    for i in range(N_ATTRS):
        attrs_raw_min, attrs_raw_max = np.min(attrs_raw[:,i]), np.max(attrs_raw[:,i])
        attrs_raw_intervals = np.linspace(start=attrs_raw_min, stop=attrs_raw_max+1e-4, num=N_LEVELS_PER_ATTR, endpoint=False)
        # subract 1 because there won't be bin 0
        attrs_quantized_oneattr = np.digitize(x=attrs_raw[:,i], bins=attrs_raw_intervals, right=False) - 1
        assert np.min(attrs_quantized_oneattr) == 0 and np.max(attrs_quantized_oneattr) == N_LEVELS_PER_ATTR-1
        attrs_quantized.append(np.reshape(attrs_quantized_oneattr, [-1,1]))
    attrs_quantized = np.concatenate(attrs_quantized, axis=1)
    assert np.shape(attrs_quantized) == np.shape(attrs_raw)
    return attrs_quantized

def _load_causal3d(args):
    ds_train, ds_test = None, None
    for split in ["train", "test"]:
        print(f"[Causal3D] loading meta-{split} data and attributes...")
        imgs_all, attrs_raw_all = [], []
        for obj_cls in trange(N_OBJ_CLS):
            img_dir = os.path.join(CAUSAL3D_DIR[split], f"images_{obj_cls}")
            attrs_raw_onecls = np.load(os.path.join(CAUSAL3D_DIR[split], f"latents_{obj_cls}.npy"))
            assert np.shape(attrs_raw_onecls) == (N_IMGS_PER_SUBDIR[split], N_ATTRS)
            for i in range(N_IMGS_PER_SUBDIR[split]):
                img_filename = f"{i:05d}.png" if split=="train" else f"{i:04d}.png"
                imgs_all.append(Image.open(os.path.join(img_dir, img_filename)).convert('RGB'))
                attrs_raw_all.append(attrs_raw_onecls[i])
        attrs_raw_all = np.array(attrs_raw_all)
        attrs_all = discretize_causal3d_attrs(attrs_raw_all)
        if split=="train":
            data_transforms = build_initial_img_transforms("meta_train", args)
            ds_train = Causal3D(imgs_all, attrs_all, data_transforms)
        else:
            data_transforms = build_initial_img_transforms("meta_test", args)
            ds_test = Causal3D(imgs_all, attrs_all, data_transforms)

    # just a placeholder
    ds_valid = ds_train

    CAUSAL3D_ATTRIBUTES_IDX_META_TRAIN = [0,1,2,6] # object location and color
    CAUSAL3D_ATTRIBUTES_IDX_META_VALID = [0,1] # without early stopping, meta validation doesn't matter
    CAUSAL3D_ATTRIBUTES_IDX_META_TEST = [7,8,9] # ground color and spotlight color and position

    # Use disjoint subset of attrs for meta splits
    metatrain_attrs_all = ds_train.attrs
    metatrain_attrs = ds_train.attrs[:,CAUSAL3D_ATTRIBUTES_IDX_META_TRAIN]
    metavalid_attrs = ds_valid.attrs[:,CAUSAL3D_ATTRIBUTES_IDX_META_VALID]
    metatest_attrs = ds_test.attrs[:,CAUSAL3D_ATTRIBUTES_IDX_META_TEST]
    metatrain_attrs_oracle = ds_train.attrs[:,CAUSAL3D_ATTRIBUTES_IDX_META_TEST]

    # generate partitions with binary classification on attributes
    metatrain_partitions_supervised = generate_attributes_based_partitions(
                                            metatrain_attrs, 
                                            N_LEVELS_PER_ATTR,
                                            'meta_train', 
                                            args)
    metavalid_partitions = generate_attributes_based_partitions(
                                            metavalid_attrs, 
                                            N_LEVELS_PER_ATTR,
                                            'meta_valid', 
                                            args)
    metatest_partitions = generate_attributes_based_partitions(
                                            metatest_attrs, 
                                            N_LEVELS_PER_ATTR,
                                            'meta_test', 
                                            args)

    metatrain_partitions_supervised_all = generate_attributes_based_partitions(
                                            metatrain_attrs_all, 
                                            N_LEVELS_PER_ATTR,
                                            'meta_train', 
                                            args)
    
    metatrain_partitions_supervised_oracle = generate_attributes_based_partitions(
                                                metatrain_attrs_oracle,
                                                N_LEVELS_PER_ATTR,
                                                'meta_train',
                                                args)

    return (
        ds_train,  
        ds_valid,  
        ds_test,   
        metatrain_partitions_supervised, 
        metatrain_partitions_supervised_all,
        metatrain_partitions_supervised_oracle, 
        metavalid_partitions,  
        metatest_partitions
    )


def load_causal3d(args):
    return _load_causal3d(args)