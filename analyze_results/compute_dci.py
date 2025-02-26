"""
Evaluate the quality of learned embeddings given a pretrained encoder.
Following metrics are evaluated:
1. Completeness 
2. Disentanglement
"""

from itertools import islice
from scipy.stats import entropy
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
import datetime
import joblib

import sys
sys.path.append("../")
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

        # normalize the encodings 
        # this is to ensure logistic regression weights are comparable in its absolute values
        # no need to normalize attributes since they are multi-class labels
        # also the logistic regression allows for bias
        latent_normalizer = StandardScaler()
        self.encodings = latent_normalizer.fit_transform(self.encodings)

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
                             shuffle=False,
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

    clf_models_path = os.path.join(CLFMODELDIR, f"clf_models_{args.encoder}_{args.dsName}.pkl")
    try:
        clf_models = joblib.load(clf_models_path)
    except FileNotFoundError:
        print(f"No trained clf model found at {clf_models_path}!")
        xs, ys = next(iter(data_loader)) 
        assert xs.shape == (n_samples, raw_latent_dim) and \
                ys.shape == (n_samples, attr_dim)
        
        print("Fitting logistic regression models...")
        clf_models = []
        for i in trange(attr_dim):
            # use lasso to encourage sparce connections
            clf = LogisticRegression(solver='saga', penalty='l1').fit(X=xs, y=ys[:,i])
            clf_models.append(clf)
        joblib.dump(clf_models, clf_models_path)
        print(f"Saved models into {clf_models_path}!")
    return metatest_ds_binded, clf_models, latent_partition
    
def collect_impt_weights(clf_models):
    clf_impt_weights = []
    for clf_model in clf_models:
        # for multi-class classification, pick the largest weight 
        # as that class is affected the most by the encoding
        clf_impt_weights_one_attr = np.max(np.abs(clf_model.coef_), axis=0)
        clf_impt_weights.append(clf_impt_weights_one_attr)
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
                clf_impt_weights_aggr_to_one_attr.append(np.mean(list(islice(impt_weights_iterator, length))))
            clf_impt_weights_aggr.append(clf_impt_weights_aggr_to_one_attr)
    return np.array(clf_impt_weights_aggr)

def compute_disentanglement_score(clf_impt_weights, args):
    print(f"Computing disentanglement score for {args.encoder}")
    latent_dim, attr_dim = np.shape(clf_impt_weights)
    # Entropy function takes care of normalizing across rows 
    entropy_vals = entropy(clf_impt_weights, base=attr_dim, axis=1)
    assert np.min(entropy_vals) >= 0 and np.max(entropy_vals) <= 1
    # compute relative waiting (to address latent dimensions that might be useless)
    relative_weights = np.sum(clf_impt_weights, axis=1) / np.sum(clf_impt_weights)
    assert np.shape(entropy_vals) == np.shape(relative_weights) == (latent_dim,)
    d_score = np.dot(1-entropy_vals, relative_weights)
    return d_score
    
def compute_completeness_score(clf_impt_weights, args):
    print(f"Computing completeness score for {args.encoder}")
    latent_dim, attr_dim = np.shape(clf_impt_weights)
    # Entropy function takes care of normalizing across columns 
    entropy_vals = entropy(clf_impt_weights, base=latent_dim, axis=0)
    assert np.min(entropy_vals) >= 0 and np.max(entropy_vals) <= 1
    assert np.shape(entropy_vals) == (attr_dim, )
    c_score = np.mean(1-entropy_vals)
    return c_score

def compute_informativeness_score(ds_binded, clf_models, args):
    print(f"Computing informativeness score for {args.encoder}")
    # take the next batch as testing points for informativeness score
    n_samples = 5_000
    data_loader = DataLoader(ds_binded,
                             batch_size=n_samples,
                             shuffle=False,
                             drop_last=False)
    data_loader_iter = iter(data_loader)
    # use the second batch to evaluate
    next(data_loader_iter) 
    xs, ys = next(data_loader_iter)
    assert xs.shape[0] == ys.shape[0] == n_samples
    raw_latent_dim, attr_dim = xs.shape[1], ys.shape[1]
    attr_pred_losses = []
    for i in trange(attr_dim):
        clf_model = clf_models[i]
        attr_pred_losses.append(log_loss(y_true=ys[:,i], y_pred=clf_model.predict(xs)))
    i_score = 1-attr_pred_losses
    return np.mean(i_score)


def compute_DCI(metatest_ds, encoder, descriptor, args):
    print(f"<<<<<<<<<<<<<<<Result Analysis for {descriptor}>>>>>>>>>>>>>>>>")
    metatest_ds_binded, clf_models, latent_partition = learn_latent_to_attribute_mapping(metatest_ds, encoder, args)
    clf_impt_weights = collect_impt_weights(clf_models)
    clf_impt_weights_aggr = aggregate_impt_weights(clf_impt_weights, latent_partition)
    d_score = compute_disentanglement_score(clf_impt_weights_aggr, args)
    c_score = compute_completeness_score(clf_impt_weights_aggr, args)
    i_score = compute_informativeness_score(metatest_ds_binded, clf_models, args)

    res_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                        "dci_res.tex")
    with open(res_filename, "a") as f:
        f.write(str(datetime.datetime.now())+'\n')
        f.write(f"{args.encoder} on {args.dsName}: \n")
        f.write(f"D: {d_score:.2f}; C: {c_score:.2f}; I: {i_score:.2f} \n")
    print(f"[Compute_completeness_scores] finished for {descriptor}!")
    return