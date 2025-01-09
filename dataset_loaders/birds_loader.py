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

[9, 15, 15, 15, 4, 15, 6, 15, 11, 15, 15, 14, 3, 15, 15, 15, 15, 5, 5, 14, 4, 4, 4, 15, 15, 15, 15, 4]
ATTRIBUTE_BINARY_CORRESPONDANCE = {
    "bill_shape": [0, 9],
    "wing_color": [9, 15],
    "upperparts_color"
}

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

def process_birds_binary_attributes(attrs, n_binaries_per_attr):
    attrs_processed = []
    start_counter = 0
    for n_binaries in n_binaries_per_attr:
        template = np.arange(n_binaries)
        attrs_processed.append(np.dot(attrs[start_counter:start_counter+n_binaries], template))
        start_counter += n_binaries
    return attrs_processed


def _load_birds(args):
    # Resize happens later in the pipeline
    data_transforms = transforms.Compose([
        transforms.ToTensor()
    ])
    birds_dir = os.path.join(DATADIR, "birds")
    n_imgs = 11788
    n_binary_attrs = 312
    n_binaries_per_attr = \
        [9, 15, 15, 15, 4, 15, 6, 15, 11, 15, 15, 14, 3, 15, 15, 15, 15, 5, 5, 14, 4, 4, 4, 15, 15, 15, 15, 4]
    assert np.sum(n_binaries_per_attr) == n_binary_attrs
    n_attrs = len(n_binaries_per_attr) # should have 28 general features
    
    print("Loading images and parsing attributes")
    imgs_all, attrs_all = [], []
    names_lines = open(os.path.join(birds_dir, "images.txt")).readlines()
    attrs_lines = open(os.path.join(birds_dir, "attributes", "image_attribute_labels.txt")).readlines()
    for i in trange(n_imgs):
        tokens = names_lines[i].split()
        img_idx, img_filename = int(tokens[0]), tokens[1]
        assert img_filename.endswith('.jpg')
        imgs_all.append(Image.open(os.path.join(birds_dir, "images", img_filename)))
        attrs_tmp = []
        for j in range(i*n_attrs, (i+1)*n_attrs):
            tokens = attrs_lines[j].split()
            assert img_idx == int(tokens[0])
            attr_exist = int(tokens[2]) # binary attribute value
            attrs_tmp.append(attr_exist)
        assert len(attrs_tmp) == n_binary_attrs
        attrs_processed = process_birds_binary_attributes(attrs_tmp, n_binaries_per_attr)
        assert len(attrs_processed) == n_attrs
        attrs_all.append(attrs_processed)
             

    # read meta split indices
    metatrain_idxs, metatest_idxs = [], []
    with open(os.path.join(birds_dir, "train_test_split.txt")) as f:
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
    metavalid_ds = metatrain_ds
    metatest_ds = Birds([imgs_all[i] for i in metatest_idxs], 
                        [attrs_all[i] for i in metatest_idxs], 
                        data_transforms)

    CELEBA_ATTRIBUTES_IDX_META_TRAIN = np.arange()
    CELEBA_ATTRIBUTES_IDX_META_VALID = np.arange(10) # without early stopping, meta validation doesn't matter
    CELEBA_ATTRIBUTES_IDX_META_TEST = np.arange(73, n_attrs)

    # Use disjoint subset of attrs for meta splits
    celeba_meta_train_attrs = celeba_meta_train_attrs_all[:,CELEBA_ATTRIBUTES_IDX_META_TRAIN]
    celeba_meta_valid_attrs = celeba_meta_valid_attrs[:,CELEBA_ATTRIBUTES_IDX_META_VALID]
    celeba_meta_test_attrs = celeba_meta_test_attrs[:,CELEBA_ATTRIBUTES_IDX_META_TEST]

    celeba_meta_train_attrs_oracle = celeba_meta_train_attrs_all[:,CELEBA_ATTRIBUTES_IDX_META_TEST]

    # generate partitions with binary classification on celeba attrs
    meta_train_partitions_supervised = generate_attributes_based_partitions(
                                            celeba_meta_train_attrs, 
                                            2,
                                            'meta_train', 
                                            args)
    meta_valid_partitions = generate_attributes_based_partitions(
                                            celeba_meta_valid_attrs, 
                                            2,
                                            'meta_valid', 
                                            args)
    meta_test_partitions = generate_attributes_based_partitions(
                                            celeba_meta_test_attrs, 
                                            2,
                                            'meta_test', 
                                            args)

    meta_train_partitions_supervised_all = generate_attributes_based_partitions(
                                            celeba_meta_train_attrs_all, 
                                            2,
                                            'meta_train', 
                                            args)
    
    meta_train_partitions_supervised_oracle = generate_attributes_based_partitions(
                                                celeba_meta_train_attrs_oracle,
                                                2,
                                                'meta_train',
                                                args)

    return (
        celeba_meta_train,  
        celeba_meta_valid,  
        celeba_meta_test,   
        meta_train_partitions_supervised, 
        meta_train_partitions_supervised_all,
        meta_train_partitions_supervised_oracle, 
        meta_valid_partitions,  
        meta_test_partitions
    )

def load_birds(args):
    return _load_birds(args)


if __name__ == "__main__":
    data_transforms = transforms.Compose([
        transforms.ToTensor()
    ])
    celeba_set = CelebA(DATADIR, 
                        split='valid', 
                        target_type='identity',
                        transform=data_transforms,
                        download=True)
    
    n_samples = 9
    img_idxs = np.random.choice(a=len(celeba_set), size=9, replace=False)
    imgs_orig = torch.stack([celeba_set[i][0] for i in img_idxs],dim=0)
    dt = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224)
    ])
    imgs_proc = torch.stack([dt(celeba_set[i][0]) for i in img_idxs],dim=0)
    dt = transforms.Resize((84,84))
    imgs_maml = torch.stack([dt(celeba_set[i][0]) for i in img_idxs],dim=0)

    os.makedirs("misc", exist_ok=True)
    save_image(imgs_orig, "misc/original_celeba_imgs.png", nrow=3)
    save_image(imgs_proc, "misc/processed_celeba_imgs.png", nrow=3)
    save_image(imgs_maml, "misc/maml_imgs.png", nrow=3)

    print("Script finished successfully!")