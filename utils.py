import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageFilter
import torch
from torchvision import transforms as T
import torchvision.transforms.functional as T_F
import os

# Hardware setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using DEVICE: {DEVICE}")
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!!!")
# meta-training setup
METATRAIN_OUTER_EPISODES = 30000 # originally 60000 episodes in cactus paper
METATRAIN_INNER_UPDATES = 5
METATEST_INNER_UPDATES = 5
NUM_TASKS_METATRAIN = 8
NUM_TASKS_METAVALID = 16
NUM_TASKS_METATEST = 1000
METATRAIN_OUTER_LR = 0.001
METATRAIN_INNER_LR = 0.05
# pre-training & fine-tuning setup
PRETRAIN_EPOCHS = 10
PRETRAIN_BATCH_SIZE = 1024
PRETRAIN_LR = 0.05
FINETUNE_STEPS = 5
FINETUNE_LR = 0.05
# Meta-GMVAE setup
GMVAE_METATRAIN_LR = 1e-4
GMVAE_BETA = 1
# Dino & deepcluster setup
NUM_ENCODING_PARTITIONS = 50
NUM_ENCODING_CLUSTERS = 300 # originally 500 in cactus paper, taking way too long
# folders for saving results
DATADIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
MODELDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trained_models")
CLFMODELDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trained_clf_models")
ENCODERDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trained_encoders")
CLUSTERDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cluster_identities")
LEARNCURVEDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_ps")
SANITYCHECKDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visualization_checks")
# The model dir should already be synched within the git repo
for dirname in [DATADIR, MODELDIR, ENCODERDIR, CLUSTERDIR, SANITYCHECKDIR]:
    os.makedirs(dirname, exist_ok=True)

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def get_descriptor(encoder, args):
    if args.dsName.startswith("mpi3d"):
        dsName_base = "mpi3d"
    elif args.dsName.startswith("celeba"):
        dsName_base = "celeba"
    else:
        dsName_base = args.dsName
    if args.encoder in ['sup', 'supora']:
        descriptor = f'{args.dsName}_{args.encoder}'
    elif args.encoder in ["supall", "scratch"]:
        # doesn't matter what attribute splits are
        descriptor = f'{dsName_base}_{args.encoder}'
    else:
        # using self-supervised/unsupervised encoder, doesn't matter what attribute splits are
        descriptor = f'{dsName_base}_{args.encoder}_{encoder.latent_dim}D_latent'
    return descriptor

def accuracy_fn(preds, labels):
    preds = preds.argmax(dim=1).view(labels.shape)
    return (preds==labels).sum().float() / labels.size(0)

def visualize_constructed_tasks(task_generator, descriptor, args, n_imgs):    
    for visual_id in range(n_imgs):
        # sample meta-training task and meta-testing task
        meta_train_task = task_generator.sample_task("meta_train", args)
        train_data, train_labels, _, test_data, test_labels, _ = meta_train_task
        grid_spacing = 0.02
        fig = plt.figure(figsize=(33+(args.NWay-1)*grid_spacing, 7.5+grid_spacing), 
                         constrained_layout=False)
        outer_grid = fig.add_gridspec(args.NWay, 2, wspace=grid_spacing, hspace=grid_spacing)

        # Iterate over classes at outer layer to ensure axis aranged according to row major
        for cls in range(args.NWay):
            support_samples = [img for (img, lbl) 
                                      in zip(train_data, train_labels)
                                        if lbl==cls]
            assert len(support_samples) == args.KShot
            inner_grid = outer_grid[cls*2].subgridspec(1, args.KShot, wspace=0.0, hspace=0.0)
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
            all_axes[i*(args.KShot+args.KQuery)].set_ylabel(f"Class {i}", fontsize=40)
        # annotate columns (at bottom of figures)
        support_col_idx = np.floor(args.KShot / 2).astype(int)
        support_ax_idx = (args.NWay-1)*(args.KShot+args.KQuery) + support_col_idx
        all_axes[support_ax_idx].set_xlabel("Support Samples", fontsize=43)
        query_col_idx = args.KShot + np.floor(args.KQuery / 2).astype(int)  
        query_ax_idx = (args.NWay-1)*(args.KShot+args.KQuery) + query_col_idx      
        all_axes[query_ax_idx].set_xlabel("Query Samples", fontsize=43)
                
        plt.savefig(os.path.join(SANITYCHECKDIR, 
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

# Croping CelebA: more strict than the cropping did in original DiTi paper. 
# Aggressive but focus on face and eliminates background noise
class CropCelebA(object):
    def __call__(self, img):
        new_img = T_F.crop(img, 57, 35, 128, 100)
        return new_img
    
class CropLFWA(object):
    def __call__(self, img):
        new_img = T_F.center_crop(img, (150, 150))
        return new_img

class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]
    
def build_initial_img_transforms(meta_split, args):
    if not args.dsNameTest:
        dsName_to_transform = args.dsName
    else:
        dsName_to_transform = args.dsName if meta_split=="meta_train" \
                                else args.dsNameTest
    # Resize happens later in the pipeline
    img_transforms = []
    if dsName_to_transform.startswith("celeba") or \
        dsName_to_transform.startswith("lfwa"):
        # for these datasets, images loaded are already in PIL format
        pass
    else:
        img_transforms.append(T.ToPILImage())
    if dsName_to_transform.startswith("celeba"):
        img_transforms.append(CropCelebA())
        img_transforms.append(T.Resize(size=(128,128)))
    elif dsName_to_transform.startswith("lfwa"):
        # make it the same size as celebA
        img_transforms.append(CropLFWA())
        img_transforms.append(T.Resize(size=(128,128)))
    if args.encoder == "simclrpretrain" and meta_split == "meta_train":
        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        img_transforms.extend([
            T.RandomResizedCrop(64, scale=(0.2, 1.0)),
            T.RandomApply(
                [T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8  # not strengthened
            ),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
            T.RandomHorizontalFlip(),
        ])
    if args.encoder == "metagmvae":
        img_transforms.append(T.Resize((args.imgSizeToEncoder, args.imgSizeToEncoder)))
    img_transforms.append(T.ToTensor())
    if dsName_to_transform == "norb":
        # turn gray-scale single channel into 3 channels
        img_transforms.append(T.Lambda(lambda x: x.repeat(3,1,1)))
    img_transforms = T.Compose(img_transforms)
    if args.encoder == "simclrpretrain" and meta_split == "meta_train":
        img_transforms=TwoCropsTransform(img_transforms)
    return img_transforms


def get_args_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dsName', 
                        help='dataset for meta-learning',
                        required=True)
    parser.add_argument('--dsNameTest',
                        help='dataset for meta-testing, if different from meta-training',
                        default=None)
    parser.add_argument('--encoder',
                        help='encoder for encodings to be clustered',
                        choices=["sup",
                                 "supall",
                                 "supora",
                                 "scratch",
                                 "simclrpretrain",
                                 "dino", 
                                 "deepcluster", 
                                 "fdae",
                                 "lsd",
                                 "metagmvae",
                                 "ablate_disentangle",
                                 "ablate_align",
                                 "ablate_individual_cluster"],
                        required=True)
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
    parser.add_argument('--KShot',
                        help='Shots for each few-shot task for meta-training',
                        type=int,
                        required=True)
    parser.add_argument('--KQuery',
                        help='number of testing samples in each task for meta-training',
                        type=int,
                        required=True)
    parser.add_argument('--KShotTest',
                        help='Shots for each few-shot task for meta-testing',
                        type=int,
                        required=True)
    parser.add_argument('--KQueryTest',
                        help='number of testing samples in each task for meta-testing',
                        type=int,
                        required=True)
    parser.add_argument('--seed',
                        help='The seed for experiment trial',
                        type=int,
                        required=True)
    parser.add_argument('--visualizeTasks',
                        help='Visualize the constructed meta-learning tasks',
                        action='store_true')
    parser.add_argument('--computePartitionOverlap',
                        help='Whether computing the partition overlap metric',
                        action='store_true')
    return parser