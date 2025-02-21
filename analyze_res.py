"""
Evaluate the quality of learned embeddings given a pretrained encoder.
Following metrics are evaluated:
1. Completeness 
2. Disentanglement
"""

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
    latent_dim_raw = encodings.shape[1]

    # bind latents and attributes
    metatest_ds_binded = BindedDataset(metatest_ds, encodings)
    n_attrs = metatest_ds_binded.get_num_attributes()
    # to avoid OOM, select a number of images for logistic regression
    n_samples = 5_000
    data_loader = DataLoader(metatest_ds_binded,
                             batch_size=n_samples, 
                             drop_last=False)
    
    xs, ys = data_loader[0] 
    assert xs.shape == (n_samples, latent_dim_raw) and \
            ys.shape == (n_samples, n_attrs)
    
    print("Fitting logistic regression models...")
    clf_models = []
    for i in trange(n_attrs):
        clf = LogisticRegression().fit(X=xs, y=ys[:,i])
        clf_models.append(clf)
    return metatest_ds_binded, clf_models
    

def compute_disentanglement_scores(clf_models, args):
    print(f"Computing disentanglement score for {args.encoder}")
    for clf_model in clf_models:

    

def compute_completeness_scores(meta_train_set, encoder, descriptor, args):

    

    with open("enc_eval.txt", "a") as f:
        f.write(str(datetime.datetime.now())+'\n')
        f.write(f"[{args.encoder} on {args.dsName}: \n")
        for i, pred_accur in enumerate(pred_accurs):
            f.write(f"on attr {i+1}: {np.mean(pred_accurs[i]):.2f}%;")
            f.write(f"std {np.std(pred_accurs[i])/np.sqrt(len(pred_accurs[i]))*100:.2f}%\n")
    print(f"[Compute_completeness_scores] finished for {descriptor}!")
    return

def compute_informativeness_scores():
    pass


def analyze_results(metatest_ds, encoder, descriptor, args):
    print(f"<<<<<<<<<<<<<<<Result Analysis for {descriptor}>>>>>>>>>>>>>>>>")
    metatest_ds_binded, clf_models = learn_latent_to_attribute_mapping(metatest_ds, encoder, args)
    compute_disentanglement_scores(clf_models, args)
    compute_completeness_scores(clf_models, args)
    compute_informativeness_scores(metatest_ds_binded, clf_models, args)
