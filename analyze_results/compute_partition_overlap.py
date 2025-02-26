import random
import os
import numpy as np
from tqdm import trange
import datetime

N_PAIRS_TO_COMPARE = 20

def _compute_joint_over_union(cluster1, cluster2):
    # create set object to use the joint and union actions
    set1, set2 = set(cluster1), set(cluster2)
    intersection = len(set1 & set2)  # `&` gives intersection
    union = len(set1 | set2)  # `|` gives union
    return intersection / union if union != 0 else 0

def _compute_partition_overlap_pairwise(partition1, partition2):
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

def compute_partition_overlap(partitions, descriptor_for_partitions):
    print(f"Compute partition overlaps for {descriptor_for_partitions}...")
    n_partitions = len(partitions)
    joint_over_union_all_pairs = []
    for i in trange(N_PAIRS_TO_COMPARE, desc="compare partition pairs..."):
        partition_idxs = random.sample(range(n_partitions), 2)
        partition1 = partitions[partition_idxs[0]]
        partition2 = partitions[partition_idxs[1]]
        joint_over_union_all_pairs.append(_compute_partition_overlap_pairwise(partition1, partition2))

    res_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                        "joint_over_union_res.txt")
    with open(res_filename, "a") as f:
        f.write(str(datetime.datetime.now())+'\n')
        f.write(f"{descriptor_for_partitions}: \n")
        f.write(f"Joint over union: avg: {np.mean(joint_over_union_all_pairs):.2f}; ")
        f.write(f"std: {np.std(joint_over_union_all_pairs):.2f} \n")
    print(f"[Compute_completeness_scores] finished for {descriptor_for_partitions}!")
    return