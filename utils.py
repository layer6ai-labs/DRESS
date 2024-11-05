import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageFilter
import torch
import os

# Hardware setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using DEVICE: {DEVICE}")
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!!!")
# meta-training setup
METATRAIN_OUTER_EPISODES = 30000 # originally 60000 episodes in cactus paper
METAVALID_OUTER_INTERVAL = 1000 # runs very rarely. currently not using early stopping. just for code integrity.
METATRAIN_INNER_UPDATES = 5
METAVALID_INNER_UPDATES = METATEST_INNER_UPDATES = 5 # reduced from original 50 updates in CACTUS paper
NUM_TASKS_METATRAIN = 8
NUM_TASKS_METAVALID = 16
NUM_TASKS_METATEST = 1000
METATRAIN_OUTER_LR = 0.001
METATRAIN_INNER_LR = 0.05
# pre-training & fine-tuning setup
PRETRAIN_EPOCHS = 10
PRETRAIN_BATCH_SIZE = 4096
PRETRAIN_LR = 0.05
FINETUNE_STEPS = 5
FINETUNE_LR = 0.05
# Meta-GMVAE setup
GMVAE_METATRAIN_LR = 1e-3
# embedding & clustering setup
NUM_ENCODING_CLUSTERS = 300 # originally 500 in cactus paper, taking way too long
# folders for saving results
DATADIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
MODELDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trained_models")
ENCODERDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trained_encoders")
CLUSTERDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cluster_identities")
LEARNCURVEDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_ps")
RESULTSDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
# The model dir should already be synched within the git repo
for dirname in [DATADIR, MODELDIR, ENCODERDIR, CLUSTERDIR, LEARNCURVEDIR]:
    os.makedirs(dirname, exist_ok=True)


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_descriptor(encoder, args):
    if args.encoder in ['sup', 'supall', 'supora', 'scratch']:
        # not using an encoder for embeddings (and clustering)
        descriptor = f'{args.dsName}_{args.encoder}' 
    else:
        descriptor = f'{args.dsName}_{args.encoder}_to_{encoder.latent_dim}D_encSpace'
    return descriptor


def visualize_constructed_tasks(task_generator, descriptor, args, n_imgs):    
    for visual_id in range(n_imgs):
        # sample meta-training task and meta-testing task
        meta_train_task = task_generator.sample_task("meta_train", args)
        train_data, train_labels, _, test_data, test_labels, _ = meta_train_task
        grid_spacing = 0.03
        fig = plt.figure(figsize=(33+(args.NWay-1)*grid_spacing, 7.5+grid_spacing), 
                         constrained_layout=False)
        outer_grid = fig.add_gridspec(args.NWay, 2, wspace=grid_spacing, hspace=grid_spacing)

        # Iterate over classes at outer layer to ensure axis aranged according to row major
        for cls in range(args.NWay):
            support_samples = [img for (img, lbl) 
                                      in zip(train_data, train_labels)
                                        if lbl==cls]
            assert len(support_samples) == args.KShotMetaTr
            inner_grid = outer_grid[cls*2].subgridspec(1, args.KShotMetaTr, wspace=0.0, hspace=0.0)
            for i, img in enumerate(support_samples):
                ax = fig.add_subplot(inner_grid[i])
                ax.imshow(torch.permute(img, (1,2,0)))
                ax.set_xticks([])
                ax.set_yticks([])
                fig.add_subplot(ax)

            query_samples = [img for (img, lbl) 
                                      in zip(test_data, test_labels)
                                        if lbl==cls]
            assert len(query_samples) == args.KQuery
            inner_grid = outer_grid[cls*2+1].subgridspec(1, args.KQuery, wspace=0.0, hspace=0.0)
            for i, img in enumerate(query_samples):
                ax = fig.add_subplot(inner_grid[i])
                ax.imshow(torch.permute(img, (1,2,0)))
                ax.set_xticks([])
                ax.set_yticks([])
                fig.add_subplot(ax)
        
        all_axes = fig.get_axes()
        # label rows
        for i in range(args.NWay):
            all_axes[i*(args.KShotMetaTr+args.KQuery)].set_ylabel(f"Class {i}", fontsize=34)
        # annotate columns (at bottom of figures)
        support_col_idx = np.floor(args.KShotMetaTr / 2).astype(int)
        support_ax_idx = (args.NWay-1)*(args.KShotMetaTr+args.KQuery) + support_col_idx
        all_axes[support_ax_idx].set_xlabel("Support Samples", fontsize=36)
        query_col_idx = args.KShotMetaTr + np.floor(args.KQuery / 2).astype(int)  
        query_ax_idx = (args.NWay-1)*(args.KShotMetaTr+args.KQuery) + query_col_idx      
        all_axes[query_ax_idx].set_xlabel("Query Samples", fontsize=36)
                
        plt.savefig(os.path.join(ENCODERDIR, 
                                 f"{descriptor}_constructed_tasks_eg{visual_id+1}.pdf"), 
                    format="pdf",
                    bbox_inches='tight')
    print(f"[visualize_constructed_tasks] finished for {descriptor}!")
    return

# Data augmentation helper classes for SIMCLR
class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


def get_args_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dsName', 
                        help='dataset for meta-learning', 
                        choices=["omniglot", 
                                 "celebarand", 
                                 "celebahard",
                                 "mpi3dtoy",
                                 "mpi3dtoyhard",
                                 "mpi3dcomplex",
                                 "mpi3dcomplexhard",
                                 "shapes3d"],
                        required=True)
    parser.add_argument('--encoder',
                        help='encoder for encodings to be clustered',
                        choices=["sup",
                                 "supall",
                                 "supora",
                                 "scratch",
                                 "simclrpretrain",
                                 "dino", 
                                 "deepcluster", 
                                 "vanillavae", 
                                 "factorvae", 
                                 "dlqvae",
                                 "fdae",
                                 "soda"],
                        required=True)
    parser.add_argument('--numEncodingPartitions',
                        help='number of partitions derived from encoding each containing classes',
                        type=int,
                        default=-1)
    parser.add_argument('--imgSizeToEncoder',
                        help='image size to encoders',
                        type=int,
                        required=True)
    parser.add_argument('--imgSizeToMetaModel',
                        help='image size to the base learner in meta-learning',
                        type=int,
                        required=True)
    parser.add_argument('--NWay',
                        help='number of classes in each classification task',
                        type=int,
                        required=True)
    parser.add_argument('--KShotMetaTr',
                        help='Shots for meta-training tasks (not used for pretrain and finetune)',
                        type=int,
                        default=5)
    parser.add_argument('--KShotMetaVa',
                        help='Shots for meta-validation tasks (not used for pretrain and finetune)',
                        type=int,
                        default=5)
    parser.add_argument('--KShotMetaTe',
                        help='Shots for meta-testing tasks',
                        type=int,
                        required=True)
    parser.add_argument('--KQuery',
                        help='number of testing samples in each task',
                        type=int,
                        required=True)
    parser.add_argument('--seed',
                        help='The seed for experiment trial',
                        type=int,
                        required=True)
    parser.add_argument('--visualizeTasks',
                        help='Visualize the constructed meta-learning tasks',
                        action='store_true')
    return parser