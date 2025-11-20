import random
import os
import numpy as np
from tqdm import trange
import datetime

N_PAIRS_TO_COMPARE = 30
PARTITION_OVERLAP_LOWERBOUND = 0.1

def _compute_joint_over_union(cluster1, cluster2):
    # create set object to use the joint and union actions
    set1, set2 = set(cluster1), set(cluster2)
    intersection = len(set1 & set2)  # `&` gives intersection
    union = len(set1 | set2)  # `|` gives union
    return intersection / union if union != 0 else 0

def _compute_partition_overlap_pairwise(partition1, partition2):
    """
    For supervised partitions, since they are created based on data point labels matching a pattern,
    each partition would only cover a subset of the entire dataset.
    For unsupervised partitions, they should initially cover the entire dataset, however we might drop
    clusters that are too small. Therefore the partition could also potentially only cover a subset.
    Before we proceed with IoU computation, we need to ensure there is sufficient shared base between
    the two partitions
    """
    partition1_all_datapoints, partition2_all_datapoints = [], []
    for cluster_key, cluster_points in partition1.items():
        partition1_all_datapoints.extend(cluster_points)
    for cluster_key, cluster_points in partition2.items():
        partition2_all_datapoints.extend(cluster_points)
    # get the joint between the subsets covered by two partitions respectively
    partition1_all_datapoints, partition2_all_datapoints = set(partition1_all_datapoints), set(partition2_all_datapoints)
    intersection_size = len(partition1_all_datapoints & partition2_all_datapoints)
    smallest_partition_size = min(len(partition1_all_datapoints), len(partition2_all_datapoints))
    assert smallest_partition_size > 0
    """
    Resample the partitions
    """
    if intersection_size / smallest_partition_size < PARTITION_OVERLAP_LOWERBOUND:
        # print("Encountering two disjoint partitions to start with, resample pairs of partitions...")
        return -1
    # for supervised partition, the keys are strings of double-digit binary values
    # also there would be k-mean clusters that have fewer than few-shot requirement and are pruned
    # therefore can't use range() to replace the keys list
    cluster_id_partition2_remained = list(partition2.keys())

    # compute the max joint over union for each cluster in partition 1 over partition 2
    joint_over_union_all = []
    for cluster_id_partition1 in partition1.keys():
        joint_over_union_max, cluster_id_partition2_max = 0, None
        for cluster_id_partition2 in cluster_id_partition2_remained:
            joint_over_union = _compute_joint_over_union(
                                    partition1[cluster_id_partition1], 
                                    partition2[cluster_id_partition2])
            if joint_over_union > joint_over_union_max:
                joint_over_union_max = joint_over_union
                cluster_id_partition2_max = cluster_id_partition2
        joint_over_union_all.append(joint_over_union_max)
        # there is a possibility that for the current cluster in partition 1, 
        # there is no cluster in partition 2 remaining with any joint 
        # then, simply don't remove any cluster from the partition 2 list
        if cluster_id_partition2_max is not None:
            cluster_id_partition2_remained.remove(cluster_id_partition2_max)

    return np.mean(joint_over_union_all)

def compute_partition_overlap(partitions, descriptor_for_partitions, args):
    print(f"Compute partition overlaps for {descriptor_for_partitions}...")
    n_partitions = len(partitions)
    joint_over_union_all_pairs = []
    for i in trange(N_PAIRS_TO_COMPARE, desc="compare partition pairs..."):
        iou_val = -1
        while iou_val < 0:
            partition_idxs = random.sample(range(n_partitions), 2)
            partition1 = partitions[partition_idxs[0]]
            partition2 = partitions[partition_idxs[1]]
            iou_val = _compute_partition_overlap_pairwise(partition1, partition2)
        joint_over_union_all_pairs.append(iou_val)

    res_filename = os.path.join(os.path.dirname(
                                    os.path.dirname(os.path.abspath(__file__))), 
                                        "res.txt")
    with open(res_filename, "a") as f:
        f.write(str(datetime.datetime.now())+'\n')
        f.write(f"IoU within meta-train {descriptor_for_partitions} on seed {args.seed}: \n")
        f.write(f"Joint over union: avg: {np.mean(joint_over_union_all_pairs):.2f} \n")
    print(f"[Compute_partition_overlap_within_metatrain] finished for {descriptor_for_partitions}!")
    return

def compute_partition_overlap_to_metatest(partitions_metatrain,
                                          partitions_metatest,
                                          descriptor_for_partitions,
                                          args):
    print(f"Compute partition overlaps for {descriptor_for_partitions}...")
    n_partitions_tr, n_partitions_te = len(partitions_metatrain), len(partitions_metatest)
    joint_over_union_all_pairs = []
    for i in trange(N_PAIRS_TO_COMPARE, desc="compare partition pairs..."):
        iou_val = -1
        while iou_val < 0:
            partition_idx_tr, partition_idx_te = random.sample(range(n_partitions_tr), 1)[0], \
                                                    random.sample(range(n_partitions_te), 1)[0]
            partition1 = partitions_metatrain[partition_idx_tr]
            partition2 = partitions_metatest[partition_idx_te]
            iou_val = _compute_partition_overlap_pairwise(partition1, partition2)
        joint_over_union_all_pairs.append(iou_val)

    res_filename = os.path.join(os.path.dirname(
                                    os.path.dirname(os.path.abspath(__file__))), 
                                        "res.txt")
    with open(res_filename, "a") as f:
        f.write(str(datetime.datetime.now())+'\n')
        f.write(f"IoU across meta-train and meta-test {descriptor_for_partitions} on seed {args.seed}: \n")
        f.write(f"Joint over union: avg: {np.mean(joint_over_union_all_pairs):.3f} \n")
    print(f"[Compute_partition_overlap_between_metatrain_metatest] finished for {descriptor_for_partitions}!")
    return