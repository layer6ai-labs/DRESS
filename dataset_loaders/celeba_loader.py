#!/usr/bin/env python3
import sys
import os
import random
from torch.utils.data import Dataset, ConcatDataset
from torchvision import transforms
from torchvision.datasets.celeba import CelebA
from torchvision.utils import save_image

sys.path.append("../")
from partition_generators import generate_attributes_based_partitions
from utils import *

def load_celeba_attrs():
    attrs_filename = os.path.join(DATADIR, "celeba", "list_attr_celeba.txt")
    splits_filename = os.path.join(DATADIR, "celeba", "list_eval_partition.txt")
    attrs_meta_train, attrs_meta_valid, attrs_meta_test = [], [], []
    with open(attrs_filename, "r") as f1, open(splits_filename, 'r') as f2:
        # skip the first two lines as headers
        for _ in range(2):
            line = f1.readline()
        print(f"Reading attributes from {attrs_filename}...")
        for attr_line, split_line in zip(f1, f2):
            attr_tokens = attr_line.strip().replace('-1', '0').split(' ')
            # remove empty strings
            attr_tokens = [attr_token for attr_token in attr_tokens if attr_token][1:]
            attrs_one_sample = np.array(attr_tokens, dtype=int)
            assert np.shape(attrs_one_sample) == (40,), \
                f"Wrong number of attributes: {np.size(attrs_one_sample)}"
            split = int(split_line.strip().split(' ')[-1])
            if split == 0:
                attrs_meta_train.append(attrs_one_sample)
            elif split == 1:
                attrs_meta_valid.append(attrs_one_sample)
            else:
                attrs_meta_test.append(attrs_one_sample)
    attrs_meta_train, attrs_meta_valid, attrs_meta_test = np.vstack(attrs_meta_train), \
                                                          np.vstack(attrs_meta_valid), \
                                                          np.vstack(attrs_meta_test)
    assert np.shape(attrs_meta_train) == (162770, 40), f"incorrect shape: {np.shape(attrs_meta_train)}" 
    assert np.shape(attrs_meta_valid) == (19867, 40), f"incorrect shape: {np.shape(attrs_meta_valid)}" 
    assert np.shape(attrs_meta_test) == (19962, 40), f"incorrect shape: {np.shape(attrs_meta_test)}" 

    print("CelebA attributes collected!")
    return attrs_meta_train, attrs_meta_valid, attrs_meta_test



def _load_celeba(args, meta_split_type):
    data_transforms = build_initial_img_transforms("meta_train", args)
    # Set up both the background and eval dataset
    celeba_meta_train = CelebA(DATADIR, 
                          split='train', 
                          target_type='attr',
                          transform=data_transforms,
                          download=True)
    
    celeba_meta_valid = CelebA(DATADIR, 
                          split='valid', 
                          target_type='attr',
                          transform=data_transforms,
                          download=True)

    data_transforms = build_initial_img_transforms("meta_test", args)    
    celeba_meta_test = CelebA(DATADIR, 
                         split='test', 
                         target_type='attr',
                         transform=data_transforms,
                         download=True)

    # collect attributes for creating supervised partitions
    celeba_meta_train_attrs_all, celeba_meta_valid_attrs, celeba_meta_test_attrs = load_celeba_attrs()

    if meta_split_type == "hair": 
        CELEBA_ATTRIBUTES_IDX_META_TEST = [5, 8, 9, 11, 17, 28, 32, 33]
    elif meta_split_type == "primary":
        CELEBA_ATTRIBUTES_IDX_META_TEST = [4, 6, 7, 9, 15, 26, 32, 35]
    elif meta_split_type == "rand":
        CELEBA_ATTRIBUTES_IDX_META_TEST = [0, 3, 4, 10, 12, 14, 16, 21]
    else:
        print(f"Invalid meta_split_type for celeba: {meta_split_type}!")
        exit(1)

    # for supervised benchmark, the labels are for attributes different from that for meta-test tasks
    CELEBA_ATTRIBUTES_IDX_META_TRAIN = [i for i in range(40) if i not in CELEBA_ATTRIBUTES_IDX_META_TEST]    

    # Use disjoint subset of attrs for meta splits
    celeba_meta_train_attrs = celeba_meta_train_attrs_all[:,CELEBA_ATTRIBUTES_IDX_META_TRAIN]
    celeba_meta_test_attrs = celeba_meta_test_attrs[:,CELEBA_ATTRIBUTES_IDX_META_TEST]

    celeba_meta_train_attrs_oracle = celeba_meta_train_attrs_all[:,CELEBA_ATTRIBUTES_IDX_META_TEST]

    # generate partitions with binary classification on celeba attrs
    meta_train_partitions_supervised = generate_attributes_based_partitions(
                                            celeba_meta_train_attrs, 
                                            2,
                                            'meta_train', 
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
        celeba_meta_test,   
        meta_train_partitions_supervised, 
        meta_train_partitions_supervised_all,
        meta_train_partitions_supervised_oracle, 
        meta_test_partitions
    )

def load_celeba_rand(args):
    return _load_celeba(args, meta_split_type='rand')

def load_celeba_hair(args):
    return _load_celeba(args, meta_split_type='hair')

def load_celeba_primary(args):
    return _load_celeba(args, meta_split_type='primary')


if __name__ == "__main__":
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        # remove margins
        transforms.Resize(256),
        transforms.CenterCrop(224)
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