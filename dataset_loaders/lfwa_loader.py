import os
import sys
import pandas as pd
from pathlib import Path
from tqdm import tqdm 
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.utils import save_image

sys.path.append("../")
from partition_generators import generate_attributes_based_partitions
from utils import *

# in total 13233 images, but only use a subset for meta-test experiments
N_LFWA_IMGS_METATEST = 2000 
N_LFWA_ATTRS = 73
RECONSTRUCT_METATEST_SET = False

class LFWA(Dataset):
    def __init__(self, df_filenames_attributes, transforms):
        self.df_filenames_attributes = df_filenames_attributes
        self.df_attributes = df_filenames_attributes.drop(columns=["filename"])
        self.transform = transforms

    def __len__(self):
        return len(self.df_filenames_attributes)

    def __getitem__(self, index):
        img_path = self.df_filenames_attributes.iloc[index]["filename"]
        img = Image.open(img_path).convert('RGB')
        attrs = self.df_attributes.iloc[index].values.astype(int)
        return (self.transform(img), torch.tensor(attrs))

def read_lfwa_attributes(attr_path: Path) -> pd.DataFrame:
    """
    lfw_attributes.txt has a header row, then tab-separated columns:
      person  imagenum  <73 attributes as float scores>
    """
    # Pandas can read tabs; the first line is a header with column names.
    df = pd.read_csv(attr_path, sep="\t", engine="python")
    assert "person" in df.columns and "imagenum" in df.columns, "Could not find 'person' and 'imagenum' columns."
    # shouldn't contain any nan or missing values
    assert not df.isnull().values.any(), "Found NaN or missing values in the attributes file."
    return df

def binarize_attributes(df_scores: pd.DataFrame, threshold: float) -> pd.DataFrame:
    # Only binarize the attribute columns (exclude 'person', 'imagenum', 'filename' if present)
    exclude = {"person", "imagenum", "filename"}
    attr_cols = [c for c in df_scores.columns if c not in exclude]
    df_bin = df_scores.copy()
    df_bin[attr_cols] = (df_scores[attr_cols].values >= threshold).astype(int)
    return df_bin

# on each computer, run this once to construct the meta-test set attributes file
def create_dataset_with_attributes(meta_split_type):
    assert meta_split_type == "meta_test"
    n_imgs = N_LFWA_IMGS_METATEST
    lfwa_attributes_filepath = os.path.join(DATADIR, "lfwa", "lfw_attributes.txt")
    df_filenames_attributes = read_lfwa_attributes(lfwa_attributes_filepath)
    # Columns: person, imagenum, then 73 attributes
    base_cols = ["person", "imagenum"]
    attr_cols = [c for c in df_filenames_attributes.columns if c not in base_cols]
    assert len(attr_cols) == N_LFWA_ATTRS, f"Expected {N_LFWA_ATTRS} attributes, found {len(attr_cols)}"
    print(f"LFWA loaded {len(attr_cols)} attributes.")

    # 4) Map each row to an actual JPG path; record misses
    img_paths, img_not_found = [], []
    for idx, row in tqdm(df_filenames_attributes.iterrows(), 
                         total=len(df_filenames_attributes), 
                         desc="LFWA: Mapping images"):
        person = str(row["person"])
        imagenum = int(row["imagenum"])
        person_dir = person.replace(" ", "_")
        img_path = os.path.join(DATADIR, 
                                "lfwa",
                                "imgs",
                                person_dir,
                                f"{person_dir}_{imagenum:04d}.jpg")
        if os.path.isfile(img_path):
            img_paths.append(img_path)
        else:
            img_paths.append(None)
            img_not_found.append((person, imagenum))
    print(f"LFWA Finished iterating image paths, with {len(img_paths)} recorded and {len(img_not_found)} not found")
    assert len(img_paths) >= n_imgs, f"Expected at least {n_imgs} images, found {len(img_paths)}"

    df_filenames_attributes["filename"] = img_paths
    df_filenames_attributes = df_filenames_attributes[df_filenames_attributes["filename"].notna()]

    # Binarize float scores from the original attribute file into (0/1) and write CSV
    df_bin = binarize_attributes(df_filenames_attributes[["filename"] + attr_cols], threshold=0)

    # sample a subset of images for meta-test
    df_bin = df_bin.sample(n=n_imgs).reset_index(drop=True)
     # save the processed attributes file
    df_bin.to_csv(os.path.join(DATADIR,
                               "lfwa",
                               f"{meta_split_type}_attributes_binary.csv"), index=False)

    print(f"LFWA: Created the {meta_split_type} set attributes with {len(df_bin)} images.")
    return

def load_lfwa(args):
    # Use significant and primary attributes for meta-testing
    # attributes selected: 
    # white, blonde hair, sunglasses, bangs, big nose, big lips, wearing hat, pale skin
    LFWA_ATTRIBUTES_IDX_META_TEST = [2, 10, 15, 29, 38, 40, 49, 63] 
    # load the pre-constructed attribute file
    meta_test_filenames_attrs = pd.read_csv(os.path.join(DATADIR,
                                               "lfwa",
                                               "meta_test_attributes_binary.csv"))
    assert len(meta_test_filenames_attrs) == N_LFWA_IMGS_METATEST, f"Expected {N_LFWA_IMGS_METATEST} images, found {len(meta_test_attrs)}"
    assert meta_test_filenames_attrs.isnull().sum().sum() == 0, "Found NaN or missing values in the attributes file."  
    data_transforms = build_initial_img_transforms(meta_split="meta_test", args=args)   
    ds_meta_test = LFWA(meta_test_filenames_attrs, data_transforms)

    meta_test_attrs = ds_meta_test.df_attributes.iloc[:, LFWA_ATTRIBUTES_IDX_META_TEST].to_numpy()
    print(f"LFWA meta-test attributes shape: {meta_test_attrs.shape}")

    meta_test_partitions = generate_attributes_based_partitions(
                                meta_test_attrs,
                                2,
                                'meta_test',
                                args)
    
    return (
        None,
        ds_meta_test,
        None,
        None,
        None,
        meta_test_partitions
    )
                    

# run below to initialize meta-split datasets
if __name__ == "__main__":
    if RECONSTRUCT_METATEST_SET:
        print("<<<<<<<<<<<<<<<<Creating a new LFWA meta-test set attributes file...>>>>>>>>>>")
        create_dataset_with_attributes("meta_test")
    
    meta_test_attrs = pd.read_csv(os.path.join(DATADIR,
                                    "lfwa",
                                    "meta_test_attributes_binary.csv"))
    assert len(meta_test_attrs) == N_LFWA_IMGS_METATEST, f"Expected {N_LFWA_IMGS_METATEST} images, found {len(meta_test_attrs)}"
    assert meta_test_attrs.isnull().sum().sum() == 0, "Found NaN or missing values in the attributes file."  
    data_transforms = transforms.Compose([
        transforms.CenterCrop(150),   
        transforms.Resize((128,128)),
        transforms.ToTensor()
    ])
    
    ds_meta_test = LFWA(meta_test_attrs, data_transforms)
    
    n_samples = 9
    img_idxs = np.random.choice(a=len(ds_meta_test), size=9, replace=False)
    imgs_orig = torch.stack([ds_meta_test[i][0] for i in img_idxs],dim=0)

    os.makedirs("misc", exist_ok=True)
    save_image(imgs_orig, "misc/lfwa_imgs.png", nrow=3)

    print("Script finished successfully!")