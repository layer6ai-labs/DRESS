import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from tqdm import trange

class DinoV2(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        assert latent_dim == 384, "testing DinoV2 raw features"
        self.dino_model = torch.hub.load(
                            'facebookresearch/dinov2', 
                            'dinov2_vits14_reg')
        self.latent_dim = latent_dim
        print("DinoV2 initialized successfully!")
   
    def encode(self, input_data: torch.Tensor) -> torch.Tensor:
        assert input_data.size(1) == 3
        with torch.no_grad():
            encodings = self.dino_model(input_data)
        assert encodings.size(1) == self.latent_dim
        return encodings
    
    def post_encode(self, encodings):
        return encodings
    
class Ablate_Disentangle(DinoV2):
    def __init__(self, latent_dim, levels_per_dim):
        super().__init__(latent_dim)
        self.levels_per_dim = levels_per_dim
        self.postencode_kmeans_model = KMeans(n_clusters=self.levels_per_dim,
                                              init='k-means++',
                                              n_init=1,
                                              max_iter=100)
        print("Ablate_Disentangle encoder initialized successfully!")

    # use k-means to create clusters on each individual latent dimension respectively
    def post_encode(self, encodings):
        print("Ablate_Disentangle start post encode...")
        dataset_size = encodings.shape[0]
        assert encodings.shape == (dataset_size, self.latent_dim)
        encodings_quantized = [self.postencode_kmeans_model.fit_predict(encodings[:,i].reshape(-1,1))
                                for i in trange(self.latent_dim)]
        encodings_quantized = np.stack(encodings_quantized, axis=1)
        encodings_quantized = torch.from_numpy(encodings_quantized) 
        assert encodings_quantized.shape == (dataset_size, self.latent_dim)
        print("Ablate_Disentangle post_encode computed successfully!")
        return encodings_quantized
