import torch
import torch.nn as nn

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
        if input_data.size(1) != 3:
            assert input_data.size(1) == 1
            # for now, just repeat the channels to get three channels
            input_data = torch.tile(input_data, (1,3,1,1))
        with torch.no_grad():
            encodings = self.dino_model(input_data)
        assert encodings.size(1) == self.latent_dim
        return encodings
    
    def post_encode(self, encodings):
        return encodings