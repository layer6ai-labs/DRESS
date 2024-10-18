"""
Evaluate the quality of learned embeddings given a pretrained encoder.
Following metrics are evaluated:
1. Completeness 
2. Disentanglement
"""

from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LogisticRegression
import datetime

from utils import *
from partition_generators import encode_data
from dataset_loaders import *
from encoders import *


EVALUATE_COMPLETENESS = False
VISUALIZE_CONSTRUCTED_TASKS = True

class BindedDataset(Dataset):
    def __init__(self, dataset, encodings):
        assert len(dataset) == len(encodings)
        self.dataset = dataset
        self.encodings = encodings

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        enc =  self.encodings[i]
        img, attr = self.dataset[i]
        return enc, attr
    

def compute_completeness_scores(meta_train_set, encoder, descriptor, args):
    encodings = encode_data(meta_train_set, encoder, args)

    # with encodings and attributes binded, can shuffle them
    data_loader = DataLoader(
                    BindedDataset(
                        meta_train_set,
                        encodings
                    ),
                  batch_size=512, 
                  shuffle=True,
                  drop_last=False)
    
    attrs_smpl = data_loader[0][1]
    pred_accurs = [[] for _ in range(attrs_smpl.shape[1])]
    for encs_batch, attrs_batch in data_loader:
        for i, attr_batch in enumerate(attrs_batch.split(1,1)):
            clf = LogisticRegression().fit(X=encs_batch, y=attr_batch.squeeze())
            pred_accur = clf.score(X=encs_batch, y=attr_batch.squeeze())
            pred_accurs[i].append(pred_accur)

    with open("enc_eval.txt", "a") as f:
        f.write(str(datetime.datetime.now())+'\n')
        f.write(f"[{args.encoder} on {args.dsName}: \n")
        for i, pred_accur in enumerate(pred_accurs):
            f.write(f"on attr {i+1}: {np.mean(pred_accurs[i]):.2f}%;")
            f.write(f"std {np.std(pred_accurs[i])/np.sqrt(len(pred_accurs[i]))*100:.2f}%\n")
    print(f"[Compute_completeness_scores] finished for {descriptor}!")
    return

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    assert min(args.NWay, args.KShotMetaTr, args.KShotMetaVa, args.KQuery) > 0
    fix_seed(args.seed)

    # load data (no need to compute the partitions)
    (
        meta_train_set, 
        meta_valid_set, 
        meta_test_set, 
        _, 
        _,
        _,
        meta_valid_partitions, 
        meta_test_partitions
    ) = LOAD_DATASET[args.dsName](args)


    encoder = get_encoder(args, DEVICE)
    descriptor = get_descriptor(encoder, args)    

    if EVALUATE_COMPLETENESS:
        compute_completeness_scores(meta_train_set, encoder, descriptor, args)
        
    
    print(f"[evaluate_encoder] {args.encoder} encodings evaluation completed!")
      