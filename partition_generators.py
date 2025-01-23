import numpy as np
from collections import defaultdict
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.cluster import KMeans
from itertools import combinations, product
from scipy.special import comb

from utils import *


'''
Supervised partition generator methods
This method is extended from the original one for celebA for allowing more than binary attributes
'''
def generate_attributes_based_partitions(attributes, code_sizes_per_attributes, meta_split, args):
    """
    Produces partitions for binary classification tasks: 
    a list of dictionaries (key: 0 or 1, value: list of data indices).
    """
    if isinstance(code_sizes_per_attributes, int):
        code_sizes_per_attributes = [code_sizes_per_attributes] * attributes.shape[1]
    assert min(code_sizes_per_attributes) > 1, "Wasteful attribute found!"
    assert attributes.shape[1] == len(code_sizes_per_attributes)
    # allowing for multiple elements indexing
    code_sizes_per_attributes = np.array(code_sizes_per_attributes)
    # Originally in CACTUS, has order of 3. After selecting a subset of more objective 
    # and noticeable features, reduce to 2 to allow for more task combinations
    order = 2
    assert order <= attributes.shape[1]

    num_partitions = 0
    partitions = []   
    n_samples_minimal = args.KShot + args.KQuery    
 
    for attr_idxs in tqdm(combinations(range(attributes.shape[1]), order), 
                          desc=f'[{args.dsName}] get_task_from_attributes', 
                          total=comb(attributes.shape[1], order)):
        code_sizes_per_attributes_subset = code_sizes_per_attributes[list(attr_idxs)]
        # ensure no repeatitive patterns generated for binary attributes (simply flipped)
        # would loss a few adjacent combinations if the first attribute is non-binary, still acceptable
        code_sizes_per_attributes_subset_for_iter = np.copy(code_sizes_per_attributes_subset)
        code_sizes_per_attributes_subset_for_iter[0] -= 1
        for pos_attr_patterns in product(*[range(code_size) for code_size in code_sizes_per_attributes_subset_for_iter]):
            neg_attr_patterns = np.mod(np.array(pos_attr_patterns) + 1, code_sizes_per_attributes_subset)
            pos_smpl_idxs = np.where(np.all([attributes[:, attr_id] == attr_val for (attr_id, attr_val) in zip(attr_idxs, pos_attr_patterns)], axis=0))[0]
            if len(pos_smpl_idxs) < n_samples_minimal:
                continue
            neg_smpl_idxs = np.where(np.all([attributes[:, attr_id] == attr_val for (attr_id, attr_val) in zip(attr_idxs, neg_attr_patterns)], axis=0))[0]
            if len(neg_smpl_idxs) < n_samples_minimal:
                continue
            # keep the sampled attribute idxs and vals as the class key instead of binary 0 and 1 class labels for visualization
            # later in task_generator, the class labels would be relabelled as 0 and 1 for meta-training
            partitions.append({f'{attr_idxs}_{np.array2string(neg_attr_patterns)}': list(neg_smpl_idxs), 
                               f'{attr_idxs}_{list(pos_attr_patterns)}': list(pos_smpl_idxs)})
            num_partitions += 1
    print(f'[{meta_split}] Generated {num_partitions} partitions by using {order}/{attributes.shape[1]} attributes')

    assert len(partitions) > 0, "At least one partition needed"
    return partitions

def generate_label_based_partition(dataset):
    partitions = []
    partition = defaultdict(list)
    for i in range(len(dataset)):
        try:
            label = dataset[i][1]
            # if label is a Tensor, then take get the scalar value
            if hasattr(label, 'item'):
                label = dataset[i][1].item()
        except ValueError as e:
            raise ValueError('Requires scalar labels. \n' + str(e))
        partition[label].append(i)
    
    # just one partition based on ground truth labels
    partitions.append(partition)
    return partitions



'''
Unsupervised partition generator methods
'''

def encode_data(dataset, encoder, args):
    assert args.imgSizeToEncoder > 0
    encode_batch_size = 1024
    if args.dsName.startswith("celeba") or args.dsName=="animals":
        if args.encoder == "FDAE":
            encode_batch_size = 32 # due to memory requirement from FDAE
        
    # simply reduce the size for the images
    data_transforms_for_encoder = transforms.Resize((
                                    args.imgSizeToEncoder, 
                                    args.imgSizeToEncoder))
        
    dl = DataLoader(dataset, 
                    batch_size=encode_batch_size,
                    shuffle=False,
                    drop_last=False)
    encodings_origSpace_tmp = []
    for data_batch, _ in tqdm(dl, desc="encoding batches"):
        encodings_origSpace_tmp.append(
            encoder.encode(data_transforms_for_encoder(data_batch).to(DEVICE)).cpu())
    encodings_origSpace_tmp = torch.concat(encodings_origSpace_tmp, dim=0)
    
    # post processing, such as PCA and kmeans
    encodings_origSpace = encoder.post_encode(encodings_origSpace_tmp)
    
    assert encodings_origSpace.shape == (len(dataset), encoder.latent_dim)
    return encodings_origSpace
 

def _diversify_encoding_spaces(encodings_origSpace: torch.Tensor, args) -> np.ndarray: 
    encodings_multiSpace = [encodings_origSpace.numpy()]
    # Following CACTUS paper, randomly scale encodings 
    for _ in range(NUM_ENCODING_PARTITIONS - 1):
        weight_vector = np.random.uniform(low=0.0, high=1.0, size=encodings_origSpace.shape[1])
        encodings_multiSpace.append(np.multiply(encodings_origSpace.numpy(), weight_vector))
    return np.stack(encodings_multiSpace, axis=0)


def _format_partition(list_of_labels, args):
    partition = defaultdict(list)
    for idx, label in enumerate(list_of_labels):
        partition[label].append(idx)
    # trim clusters with insufficient number of samples 
    # this part should only be invoked for meta-training
    labels_to_prune = [label for label, idxs in partition.items() 
                       if len(idxs) < args.KShot+args.KQuery]
    for label in labels_to_prune:
        del partition[label]
    # ensure there is enough classes with sufficient samples
    if len(partition) < args.NWay:
        print("Getting a partition with insufficient clusters: ", partition.keys())
        return None 
    return partition

    
# '''
# Only called on meta-train split
# Load the pre-computed cluster results if exists
# '''
def generate_unsupervised_partitions(
        dataset, 
        encoder,  
        descriptor, 
        args):
    cluster_idxs_filename = os.path.join(CLUSTERDIR, f"{descriptor}_clusterIdxs.npy")
    try:
        cluster_idxs = np.load(cluster_idxs_filename)
        print(f"[{descriptor}] Pre-computed cluster identities loaded!")
    except FileNotFoundError:
        print(f"[{descriptor}] No pre-computed clusters exist. Compute from beginning...")
        encodings_origSpace = encode_data(dataset, encoder, args)

        if args.encoder in ["factorvae", "fdae", "dlqvae", "diti"]:
            n_partitions = encoder.latent_dim
            # simply use the index of the quantized latent code as the cluster identity
            # and use different latent dimension as different partitions
            cluster_idxs = torch.transpose(encodings_origSpace, 0, 1).numpy()   
        else:
            n_partitions = NUM_ENCODING_PARTITIONS            
            encodings_multiSpaces = _diversify_encoding_spaces(encodings_origSpace, args)

            # for each encoding space, generate a partition through k-means clustering
            cluster_idxs = []
            print(f"[{descriptor}] Clustering and collecting cluster identities...")
            for encoding in tqdm(encodings_multiSpaces, desc="KMeans computation"):
                while True:
                    # have n_init=1, and therefore n_jobs wouldn't help in accelerating
                    kmeans = KMeans(n_clusters=NUM_ENCODING_CLUSTERS, 
                                    init='k-means++', 
                                    n_init=1, 
                                    max_iter=100).fit(encoding)
                    uniques, counts = np.unique(kmeans.labels_, return_counts=True)
                    num_big_enough_clusters = np.sum(counts > (args.KShot+args.KQuery))
                    if num_big_enough_clusters > args.NWay * 3:
                        break
                    else:
                        tqdm.write("Too few classes ({}) with greater than {} examples.".format(
                                    num_big_enough_clusters, args.KShot+args.KQuery))
                        tqdm.write('Frequency: {}'.format(counts))
                assert max(kmeans.labels_)+1 == NUM_ENCODING_CLUSTERS
                cluster_idxs.append(np.array(kmeans.labels_))
            cluster_idxs = np.stack(cluster_idxs, axis=0)
        
        assert np.shape(cluster_idxs) == (n_partitions, len(dataset))
        np.save(cluster_idxs_filename, cluster_idxs)
        print(f"[{descriptor}] Cluster identities computed and saved at {cluster_idxs_filename}!")

    assert np.shape(cluster_idxs)[1] == len(dataset), f"Wrong shape: {np.shape(cluster_idxs)}"
    unsupervised_partitions = []
    for cluster_idxs_one_partition in cluster_idxs:
        partition = _format_partition(cluster_idxs_one_partition, args)
        if not partition:
            continue
        unsupervised_partitions.append(partition)
    print(f"{descriptor}: {len(unsupervised_partitions)} unsupervised partitions generated!")
    return unsupervised_partitions
