#!/usr/bin/env python3
import sys
import os
import random
from torch.utils.data import Dataset, ConcatDataset
from torchvision import transforms
from torchvision.datasets.celeba import CelebA
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm, trange

sys.path.append("../")
from partition_generators import generate_attributes_based_partitions
from utils import *

N_IMGS = 11788
N_BINARY_ATTRS = 312
ATTR_BINARY_CORRESPONDENCE = [
    [0, 9], # bill_shape f0
    [9, 15], # wing_color f1
    [24, 15], # upperparts_color f2
    [39, 15], # underparts_color f3
    [54, 4], # breast pattern f4
    [58, 15], # back_color f5
    [73, 6], # tail_shape f6
    [79, 15], # upper_tail_color f7
    [94, 11], # head_pattern f8
    [105, 15], # breast_color f9
    [120, 15], # throat_color f10
    [135, 14], # eye_color f11
    [149, 3], # bill_length f12
    [152, 15], # forehead_color f13
    [167, 15], # undertail_color f14
    [182, 15], # nape_color f15
    [197, 15], # belly_color f16
    [212, 5], # wing_shape f17
    [217, 5], # size f18
    [222, 14], # shape f19
    [236, 4], # back_pattern f20
    [240, 4], # tail_pattern f21
    [244, 4], # belly_pattern f22
    [248, 15], # primary_color f23
    [263, 15], # leg_color f24
    [278, 15], # bill_color f25
    [293, 15], # crown_color f26
    [308, 4], # wing_pattern f27
]
ATTR_BINARY_COUNTS = np.array([i[1] for i in ATTR_BINARY_CORRESPONDENCE])
assert np.sum(ATTR_BINARY_COUNTS) == N_BINARY_ATTRS 
N_ATTRS = len(ATTR_BINARY_CORRESPONDENCE) # should have 28 processed attributes

BIRDS_DIR = os.path.join(DATADIR, "birds")


class Birds(Dataset):
    def __init__(self, imgs, attrs, transforms):
        assert len(imgs) == len(attrs)
        self.imgs = imgs
        self.attrs = attrs
        self.transforms = transforms

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        return (self.transforms(self.imgs[index]), torch.tensor(self.attrs[index]))

def process_birds_binary_attributes(attrs):
    attrs_processed = []
    for start_id, n_binaries in ATTR_BINARY_CORRESPONDENCE:
        template = np.arange(n_binaries)
        attrs_processed.append(np.dot(attrs[start_id:start_id+n_binaries], template))
    return attrs_processed

def _load_birds(args):
    # Resize happens later in the pipeline
    data_transforms = transforms.Compose([
        transforms.ToTensor()
    ])
    
    print("Loading images and parsing attributes")
    imgs_all, attrs_all = [], []
    names_lines = open(os.path.join(BIRDS_DIR, "images.txt")).readlines()
    attrs_lines = open(os.path.join(BIRDS_DIR, "attributes", "image_attribute_labels.txt")).readlines()
    for i in trange(N_IMGS):
        tokens = names_lines[i].split()
        img_idx, img_filename = int(tokens[0]), tokens[1]
        assert img_filename.endswith('.jpg')
        imgs_all.append(Image.open(os.path.join(BIRDS_DIR, "images", img_filename)))
        attrs_tmp = []
        for j in range(i*N_BINARY_ATTRS, (i+1)*N_BINARY_ATTRS):
            tokens = attrs_lines[j].split()
            assert img_idx == int(tokens[0])
            attr_exist = int(tokens[2]) # binary attribute value
            attrs_tmp.append(attr_exist)
        assert len(attrs_tmp) == N_BINARY_ATTRS
        attrs_processed = process_birds_binary_attributes(attrs_tmp)
        assert len(attrs_processed) == N_ATTRS
        attrs_all.append(attrs_processed)
             

    # read meta split indices
    metatrain_idxs, metatest_idxs = [], []
    with open(os.path.join(BIRDS_DIR, "train_test_split.txt")) as f:
        for split_line in f:
            tokens = split_line.split()
            img_idx, split_idx = int(tokens[0]), int(tokens[1])
            if split_idx == 0:
                # images are 1-indexed while here indexing into lists
                metatrain_idxs.append(img_idx-1)
            else:
                metatest_idxs.append(img_idx-1)

    metatrain_ds = Birds([imgs_all[i] for i in metatrain_idxs], 
                         [attrs_all[i] for i in metatrain_idxs], 
                         data_transforms)
    metatrain_attrs_all = metatrain_ds.attrs
    metavalid_ds = metatrain_ds
    metavalid_attrs = metatrain_attrs_all
    metatest_ds = Birds([imgs_all[i] for i in metatest_idxs], 
                        [attrs_all[i] for i in metatest_idxs], 
                        data_transforms)
    metatest_attrs = metatest_ds.attrs

    ATTRS_IDX_METATRAIN = [1,2,3,4,5,9,17,18,19,22,27]
    ATTRS_IDX_METAVALID = [1,2,3] # without early stopping, meta validation doesn't matter
    ATTRS_IDX_METATEST = [0,6,7,8,12,21,25]

    # Use disjoint subset of attrs for meta splits
    metatrain_attrs = metatrain_attrs_all[:,ATTRS_IDX_METATRAIN]
    metavalid_attrs = metavalid_attrs[:,ATTRS_IDX_METAVALID]
    metatest_attrs = metatest_attrs[:,ATTRS_IDX_METATEST]

    metatrain_attrs_oracle = metatrain_attrs_all[:,ATTRS_IDX_METATEST]

    # generate partitions with binary classification on celeba attrs
    metatrain_partitions_supervised = generate_attributes_based_partitions(
                                            metatrain_attrs, 
                                            ATTR_BINARY_COUNTS[ATTRS_IDX_METATRAIN],
                                            'meta_train', 
                                            args)
    metavalid_partitions = generate_attributes_based_partitions(
                                            metavalid_attrs, 
                                            ATTR_BINARY_COUNTS[ATTRS_IDX_METAVALID],
                                            'meta_valid', 
                                            args)
    metatest_partitions = generate_attributes_based_partitions(
                                            metatest_attrs, 
                                            ATTR_BINARY_COUNTS[ATTRS_IDX_METATEST],
                                            'meta_test', 
                                            args)
    metatrain_partitions_supervised_all = generate_attributes_based_partitions(
                                            metatrain_attrs_all, 
                                            ATTR_BINARY_COUNTS,
                                            'meta_train', 
                                            args)
    
    metatrain_partitions_supervised_oracle = generate_attributes_based_partitions(
                                                metatrain_attrs_oracle,
                                                ATTR_BINARY_COUNTS[ATTRS_IDX_METATEST],
                                                'meta_train',
                                                args)

    return (
        metatrain_ds,  
        metavalid_ds,  
        metatest_ds,   
        metatrain_partitions_supervised, 
        metatrain_partitions_supervised_all,
        metatrain_partitions_supervised_oracle, 
        metavalid_partitions,  
        metatest_partitions
    )

def load_birds(args):
    return _load_birds(args)


if __name__ == "__main__":
    names_lines = open(os.path.join(BIRDS_DIR, "images.txt")).readlines()
    n_samples = 16
    img_idxs = np.random.choice(a=N_IMGS, size=n_samples, replace=False)
    imgs_orig, imgs_maml = [], []
    for idx in img_idxs:
        tokens = names_lines[idx].split()
        img_idx, img_filename = int(tokens[0]), tokens[1]
        assert img_filename.endswith('.jpg')
        img_raw = Image.open(os.path.join(BIRDS_DIR, 'images', img_filename))
        dt_orig = transforms.Compose([transforms.ToTensor(), 
                                      transforms.Resize((224,224))])
        imgs_orig.append(dt_orig(img_raw))
        dt_maml = transforms.Compose([transforms.ToTensor(), transforms.Resize((84, 84))])
        imgs_maml.append(dt_maml(img_raw))
    imgs_orig, imgs_maml = torch.stack(imgs_orig, dim=0), torch.stack(imgs_maml, dim=0)

    os.makedirs("misc", exist_ok=True)
    save_image(imgs_orig, "misc/original_birds_imgs.png", nrow=4)
    save_image(imgs_maml, "misc/maml_birds_imgs.png", nrow=4)

    print("Script finished successfully!")