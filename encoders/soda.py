import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet

import sys
sys.path.append("../")
from utils import *

"""
Code adapted from https://github.com/FutureXiang/soda
"""

class Encoder_Network(nn.Module):
    """
    An encoder network (image -> feature_dim)
    """
    def __init__(self, feature_dim):
        super(Encoder_Network, self).__init__()

        resnet_arch = getattr(resnet, 'resnet18')
        net = resnet_arch(num_classes=feature_dim)

        self.encoder = []
        for name, module in net.named_children():
            if isinstance(module, nn.Linear):
                self.encoder.append(nn.Flatten(1))
                self.encoder.append(module)
            else:
                self.encoder.append(module)
        self.encoder = nn.Sequential(*self.encoder)

    def forward(self, x):
        return self.encoder(x)
    
class SODA(nn.Module):
    def __init__(self, latent_dim, levels_per_dim, args):
        super(SODA, self).__init__()    
        self.latent_dim = latent_dim
        self.levels_per_dim = levels_per_dim
        self.encoder = Encoder_Network(self.latent_dim)
        # load the checkpoint from SODA training run
        checkpoint = torch.load(os.path.join(ENCODERDIR, f"soda_{args.dsName}.pth"))['MODEL']
        # load the trained SODA encoder model
        self.encoder.load_state_dict(checkpoint, strict=False)

        # preprocess the celebA images as when SODA is trained
        self.data_transform = transforms.Normalize(
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]
                              )
        print(f"SODA encoder trained for {args.dsName} constructed and loaded successfully!")    

    def encode(self, input_data: torch.Tensor) -> torch.Tensor:
        assert input_data.size(1) == 3, "SODA only takes input with 3 channels!"
        input_data = self.data_transform(input_data)
        with torch.no_grad():
            encodings = self.encoder(input_data)
        assert encodings.size(1) == self.latent_dim  
        return encodings
    
    # ATTENTION: This function expects the entire dataset as input
    # Therefore the input encodings are not put into CUDA device
    def post_encode(self, encodings):
        data_size, n_features = encodings.shape
        assert n_features == self.latent_dim
        level_interval = data_size / self.levels_per_dim
        encodings_order_tmp = torch.argsort(encodings, dim=0)
        encodings_order = torch.empty(encodings.shape, dtype=torch.long).scatter_(
            dim=0,
            index=encodings_order_tmp,
            src=torch.arange(data_size).unsqueeze(1).repeat(1, n_features)
        )
        # quantize using ranking percentiles
        encodings_quantized = torch.floor(encodings_order / level_interval)
        assert encodings_quantized.max() == self.levels_per_dim - 1
        return encodings_quantized
    
