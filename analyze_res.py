"""
Evaluate the quality of learned embeddings given a pretrained encoder.
Following metrics are evaluated:
1. Completeness 
2. Disentanglement
"""

from itertools import islice
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
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
        self.encodings = encodings.numpy()
        self.attributes = np.stack([attr for _, attr in self.dataset], axis=0)
        self.raw_latent_dim = np.shape(self.encodings)[1]
        self.attr_dim = np.shape(self.attributes)[1]
        assert np.shape(self.encodings) == (len(dataset), self.raw_latent_dim)
        assert np.shape(self.attributes) == (len(dataset), self.attr_dim)

        # normalize both encodings and attributes
        # this is to ensure logistic regression weights are comparable in its absolute values
        latent_normalizer, attr_normalizer = StandardScaler(), StandardScaler()
        self.encodings = latent_normalizer.fit_transform(self.encodings)
        self.attributes = attr_normalizer.fit_transform(self.attributes)


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        return self.encodings[i], self.attributes[i]
    
def learn_latent_to_attribute_mapping(metatest_ds, encoder, args):
    encodings = encode_data(metatest_ds, encoder, args, return_raw_encodings=True)
    assert len(metatest_ds) == encodings.shape[0]
    if args.encoder in ["fdae", "lsd"]:
        encodings = encodings.reshape(encodings.shape[0], -1)
    assert encodings.ndim == 2, f"there shouldn't be another encoder with more than 1 dimensional encodings: {args.encoder}"

    # bind latents and attributes
    metatest_ds_binded = BindedDataset(metatest_ds, encodings)
    raw_latent_dim, attr_dim = metatest_ds_binded.raw_latent_dim, metatest_ds_binded.attr_dim
    # to avoid OOM, select a number of images for logistic regression
    n_samples = 5_000
    data_loader = DataLoader(metatest_ds_binded,
                             batch_size=n_samples, 
                             drop_last=False)
    # get latent partitions based on encoder
    if args.encoder == "dino":
        assert raw_latent_dim == 384
        latent_partition = [1]*384
    elif args.encoder == "deepcluster":
        assert raw_latent_dim == 256
        latent_partition = [1]*256
    elif args.encoder == "fdae":
        assert raw_latent_dim == encoder.latent_dim * 2 * encoder._code_length
        latent_partition = [encoder._code_length] * (encoder.latent_dim * 2)
    elif args.encoder == "lsd":
        assert raw_latent_dim == encoder.latent_dim * encoder.dim_per_slot
        latent_partition = [encoder.dim_per_slot] * encoder.latent_dim
    else:
        print(f"Unimplemented analysis for encoder: {args.encoder}!")
        exit(1)
    assert sum(latent_partition) == raw_latent_dim

    xs, ys = data_loader[0] 
    assert xs.shape == (n_samples, raw_latent_dim) and \
            ys.shape == (n_samples, attr_dim)
    
    print("Fitting logistic regression models...")
    clf_models = []
    for i in trange(attr_dim):
        # use lasso to encourage sparce connections
        clf = LogisticRegression(penalty='l1').fit(X=xs, y=ys[:,i])
        clf_models.append(clf)
    return metatest_ds_binded, clf_models, latent_partition
    
def collect_impt_weights(clf_models):
    clf_impt_weights = []
    for clf_model in clf_models:
        clf_impt_weights.append(clf_model.coef_)
    return clf_impt_weights

def aggregate_impt_weights(clf_impt_weights, latent_partition):
    clf_impt_weights_aggr = []
    for clf_impt_weights_to_one_attr in clf_impt_weights:
        if max(latent_partition) == 1:
            clf_impt_weights_aggr.append(clf_impt_weights_to_one_attr)
        else:
            clf_impt_weights_aggr_to_one_attr = []
            impt_weights_iterator = iter(clf_impt_weights_to_one_attr)
            for length in latent_partition:
                clf_impt_weights_aggr_to_one_attr.append(np.mean(islice(impt_weights_iterator, length)))
            clf_impt_weights_aggr.append(clf_impt_weights_aggr_to_one_attr)
    return np.array(clf_impt_weights_aggr)

def compute_disentanglement_scores(clf_impt_weights, args):
    print(f"Computing disentanglement score for {args.encoder}")
    return
    

def compute_completeness_scores(clf_impt_weights, args):
    print(f"Computing completeness score for {args.encoder}")
    return


def compute_informativeness_scores(ds_binded, clf_models, args):
    print(f"Computing informativeness score for {args.encoder}")
    return


def analyze_results(metatest_ds, encoder, descriptor, args):
    print(f"<<<<<<<<<<<<<<<Result Analysis for {descriptor}>>>>>>>>>>>>>>>>")
    metatest_ds_binded, clf_models, latent_partition = learn_latent_to_attribute_mapping(metatest_ds, encoder, args)
    clf_impt_weights = collect_impt_weights(clf_models)
    clf_impt_weights_aggr = aggregate_impt_weights(clf_impt_weights, latent_partition)
    d_score = compute_disentanglement_scores(clf_impt_weights_aggr, latent_partition, args)
    c_score = compute_completeness_scores(clf_impt_weights_aggr, latent_partition, args)
    i_score = compute_informativeness_scores(metatest_ds_binded, clf_models, args)

    with open("enc_eval.txt", "a") as f:
        f.write(str(datetime.datetime.now())+'\n')
        f.write(f"[{args.encoder} on {args.dsName}: \n")
        for i, pred_accur in enumerate(pred_accurs):
            f.write(f"on attr {i+1}: {np.mean(pred_accurs[i]):.2f}%;")
            f.write(f"std {np.std(pred_accurs[i])/np.sqrt(len(pred_accurs[i]))*100:.2f}%\n")
    print(f"[Compute_completeness_scores] finished for {descriptor}!")
    return