import sys
sys.path.append("../")
from utils import *
from partition_generators import generate_supervised_partitions_single_attribute

from torchvision.datasets import Omniglot

def load_omniglot(args):
    # meta-training set
    data_transforms = build_initial_img_transforms(meta_split="meta_train", args=args)
    meta_train_ds = Omniglot(DATADIR,
                            background=True,   # True = background set, False = evaluation set
                            download=True,
                            transform=data_transforms
                            )
    # meta-testing set
    # Eval labels also start from 0.
    # It's important to add 964 to label values in eval so they don't overwrite background dataset.
    data_transforms = build_initial_img_transforms(meta_split="meta_test", args=args)
    meta_test_ds = Omniglot(DATADIR,
                            background=False,   # True = background set, False = evaluation set
                            download=True,
                            transform=data_transforms,
                            target_transform=lambda x: x+len(meta_train_ds._characters)
                            )
    
    # simply use omniglot labels for generating tasks
    (
        meta_train_partitions_supervised, 
        meta_test_partitions
    ) = generate_supervised_partitions_single_attribute([label for img, label in meta_train_ds]), \
            generate_supervised_partitions_single_attribute([label for img, label in meta_test_ds])

    # there is no common labels among meta splits, so for SupAll and SupOra there 
    # is no corresponding labels and partitions
    return (
        meta_test_ds, 
        meta_test_ds,  
        meta_train_partitions_supervised,  
        None,
        None,
        meta_test_partitions  
    )

