#!/usr/bin/env python3

"""
General wrapper to help create tasks.
"""

import random
from torchvision import transforms

from utils import *

class TaskGenerator():
    def __init__(self, 
                 meta_train_set, 
                 meta_valid_set, 
                 meta_test_set,
                 meta_train_partitions,
                 meta_valid_partitions,
                 meta_test_partitions,
                 args):
        self.meta_train_set = meta_train_set
        self.meta_valid_set = meta_valid_set
        self.meta_test_set = meta_test_set
        self.meta_train_partitions = meta_train_partitions
        self.meta_valid_partitions = meta_valid_partitions
        self.meta_test_partitions = meta_test_partitions
        self.image_resize_transforms = transforms.Resize((args.imgSizeToMetaModel, args.imgSizeToMetaModel))
        

    def _sample_task_idxs_labels(self, partition, meta_split, args):
        if meta_split == "meta_train":
            n_train_samples = args.KShotMetaTr  
        elif meta_split == "meta_valid":
            n_train_samples = args.KShotMetaVa
        elif meta_split == "meta_test":
            n_train_samples = args.KShotMetaTe
        else:
            print(f"Invalid split: {meta_split}!")
            exit(1)    

        (
            train_idxs, 
            train_labels, 
            train_labels_orig, 
            test_idxs, 
            test_labels,
            test_labels_orig
         ) = [], [], [], [], [], []
        clses_in_partition = list(partition.keys())
        assert len(clses_in_partition) >= args.NWay, \
            f"On {meta_split}, partition has {len(clses_in_partition)} cls, while expecting {args.NWay}"
        sampled_clses = random.sample(clses_in_partition, args.NWay)
        random.shuffle(sampled_clses)
        for label, cls in enumerate(sampled_clses):
            idxs = random.sample(partition[cls], n_train_samples+args.KQuery)
            train_idxs.extend(idxs[:n_train_samples])
            train_labels.extend([label for _ in range(n_train_samples)])
            train_labels_orig.extend([cls for _ in range(n_train_samples)])
            test_idxs.extend(idxs[n_train_samples:])
            test_labels.extend([label for _ in range(args.KQuery)])
            test_labels_orig.extend([cls for _ in range(args.KQuery)])
        return train_idxs, train_labels, train_labels_orig, test_idxs, test_labels, test_labels_orig
    
    def _sample_partition(self, partitions):
        # If there is only one partition, always return this partition for constructing all tasks
        # which would be the case for supervised task construction
        assert len(partitions) > 0
        return partitions[np.random.choice(len(partitions))]
    
    def sample_task(self, meta_split, args):
        if meta_split == 'meta_train':
            partition_for_task = self._sample_partition(self.meta_train_partitions)
        elif meta_split == 'meta_valid':
            partition_for_task = self._sample_partition(self.meta_valid_partitions)
        elif meta_split == "meta_test":
            partition_for_task = self._sample_partition(self.meta_test_partitions)
        else:
            print(f"Invalid argument {meta_split}!")
            exit(1)

        # sample the labels as true labels, idxs are idxs in filtered metadataset
        train_idxs, train_labels, train_labels_orig, test_idxs, test_labels, test_labels_orig =  \
            self._sample_task_idxs_labels(partition_for_task, meta_split, args)
        # use idxs in filtered metadataset to index, which intrinsically would return the intended sample
        if meta_split=="meta_train":
            meta_set_to_gather = self.meta_train_set
        elif meta_split=="meta_valid":
            meta_set_to_gather = self.meta_valid_set
        else:
            meta_set_to_gather = self.meta_test_set
            
        # could potentially use torch.utils.data.Subset here, however we want to extract images only
        train_data = torch.stack([meta_set_to_gather[id][0] for id in train_idxs], dim=0)
        test_data = torch.stack([meta_set_to_gather[id][0] for id in test_idxs], dim=0)

        # apply resize transformation for input of base learner in meta-learning 
        assert args.imgSizeToMetaModel > 0
        train_data, test_data = self.image_resize_transforms(train_data), \
                                    self.image_resize_transforms(test_data)
        train_labels, test_labels = torch.tensor(train_labels), torch.tensor(test_labels)
        return train_data, train_labels, train_labels_orig, test_data, test_labels, test_labels_orig
