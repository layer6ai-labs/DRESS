import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

sys.path.append("../")
from utils import *


def construct_vae_encoder(conv_params, latent_dim, fc_hidden_dim, img_size):
    in_dim = 3
    encoder_conv_list = []
    for (n_chnl, kn_size, strd, pad) in conv_params:
        encoder_conv_list.append(
            nn.Sequential(
                nn.Conv2d(in_channels=in_dim, 
                          out_channels=n_chnl,
                          kernel_size=kn_size, 
                          stride=strd, 
                          padding=pad),
                nn.BatchNorm2d(n_chnl),
                nn.LeakyReLU())
        )
        in_dim = n_chnl
    encoder_conv_lyrs = nn.Sequential(*encoder_conv_list)
    # throw in a pseudo image to see the convolution layers output size
    dummy_out = encoder_conv_lyrs(torch.rand(1, 3, img_size, img_size))
    conv_out_size = dummy_out.shape[2]
    # add in flattened fully connected layers
    encoder_fc_mu = nn.Sequential(
        nn.Linear(conv_params[-1][0] * conv_out_size**2, fc_hidden_dim),
        nn.LeakyReLU(),
        nn.Linear(fc_hidden_dim, latent_dim)
    )
    return encoder_conv_lyrs, encoder_fc_mu, conv_out_size

"""
Code adapted and modified on top of
https://github.com/lucidrains/vector-quantize-pytorch
Disentanglement via Latent Quantization
 - https://arxiv.org/abs/2305.18378
"""
class LatentQuantizer(nn.Module):
    def __init__(
        self,
        latent_dim,
        levels_per_dim
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.levels_per_dim = levels_per_dim

        # ensure zero is in the middle and start is always -0.5
        values_per_latent = [
            torch.linspace(-0.5, 0.5, self.levels_per_dim) if self.levels_per_dim % 2 == 1 \
                                else torch.arange(self.levels_per_dim) / self.levels_per_dim - 0.5
            for _ in range(self.latent_dim)
        ]

        # Add the values per latent into the model parameters for optimization
        self.values_per_latent = nn.ParameterList(
            [nn.Parameter(values) for values in values_per_latent]
        )
            

    def compute_latent_quant_loss(self, z: Tensor, zhat: Tensor) -> Tensor:
        """Computes the quantization loss."""
        return F.mse_loss(zhat.detach(), z, reduction="mean")

    def compute_latent_commit_loss(self, z: Tensor, zhat: Tensor) -> Tensor:
        """Computes the commitment loss."""
        return F.mse_loss(z.detach(), zhat, reduction="mean")

    def quantize(self, z: Tensor) -> Tensor:
        """Quantizes z, returns quantized zhat, same shape as z.
        The quantization is done by measuring the distance between the input and the codebook values per latent dimension
        and returning the index of the closest codebook value.
        """

        def distance(x, y):
            return torch.abs(x - y)

        quant_idxs = torch.stack(
            [
                torch.argmin(distance(z[:, i].view(-1,1), self.values_per_latent[i]), dim=1)
                    for i in range(self.latent_dim)
            ],
            dim=-1,
        )
        z_quant = torch.stack(
            [
                self.values_per_latent[i][quant_idxs[:, i]]
                for i in range(self.latent_dim)
            ],
            dim=-1,
        )
        
        return z_quant, quant_idxs


    def forward(self, z: Tensor) -> Tensor:
        assert (z.shape[-1] == self.latent_dim), f"{z.shape[-1]} VS {self.latent_dim}"
        z_quant, quant_idxs = self.quantize(z)

        # compute the two-part latent loss here, before cutting off 
        # the gradients to the latent code
        loss_quant = self.compute_latent_quant_loss(z, z_quant)
        loss_commit = self.compute_latent_commit_loss(z, z_quant)

        # This code is brought outsize the quantize(), later to here
        # preserve the gradients on z for reconstruction loss via the 
        # straight-through gradient estimator
        # however this would cut off the gradient of the z_quant
        z_quant_for_recon = z + (z_quant - z).detach()

        return z_quant_for_recon, quant_idxs, loss_quant, loss_commit


class VAE(nn.Module):
    def __init__(self, latent_dim, args):
        super(VAE, self).__init__()
        self.model_type = args.encoder
        self.conv_params = [(32, 2, 1, 0), 
                            (64, 3, 1, 0), 
                            (128, 3, 1, 0), 
                            (256, 4, 1, 1),
                            (128, 4, 2, 1),
                            (64, 4, 2, 1)]
        self.latent_dim = latent_dim
        self.img_size = args.imgSizeToEncoder
        self.fc_hidden_dim = 256
        (
            self.encoder_conv_lyrs, 
            self.encoder_fc_mu,  
            self.conv_out_size
        ) = construct_vae_encoder(self.conv_params, 
                                  self.latent_dim,
                                  self.fc_hidden_dim,
                                  self.img_size)

        print(f"Constructed {self.model_type}, with output after convolution layers: {self.conv_params[-1][0]}X{self.conv_out_size}X{self.conv_out_size}")
        # load trained checkpoint
        checkpoint = torch.load(os.path.join(ENCODERDIR, f'{self.model_type}_{args.dsName}.pth'))['model']
        # delete the decoder weights as not needed here
        lyr_keys = list(checkpoint.keys())
        for key in lyr_keys:
            if 'decoder' in key or 'encoder_fc_var' in key:
                del checkpoint[key]
        self.load_state_dict(checkpoint)        
        print(f"{self.model_type} trained for {args.dsName} constructed and loaded successfully!")


    def forward(self, x):
        z = self.encoder_conv_lyrs(x)
        z = torch.flatten(z, start_dim=1)
        mu = self.encoder_fc_mu(z)
        # return mu as the corresponding latent vector
        return mu
    
    def encode(self, input_data: torch.Tensor) -> torch.Tensor:
        assert input_data.shape[1:] == (3, self.img_size, self.img_size), \
                    f"Incorrect input image shape for VAE: {input_data.shape}"
        with torch.no_grad():
            encodings = self.forward(input_data)
        assert encodings.shape == (input_data.size(0), self.latent_dim)

        return encodings
    
    def post_encode(self, encodings):
        return encodings
    
class FactorVAE(VAE):
    def __init__(self, latent_dim, levels_per_dim, args):
        super(FactorVAE, self).__init__(latent_dim, args)
        self.levels_per_dim = levels_per_dim

    # ATTENTION: This function expects the entire dataset as input
    # Therefore the input encodings are not put into device
    def post_encode(self, encodings):
        data_size, n_features = encodings.shape
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
    

class DLQVAE(nn.Module):
    def __init__(self, latent_dim_before_quant, latent_dim, levels_per_dim, args):
        super(DLQVAE, self).__init__()
        self.conv_params = [(64, 3, 1, 0), 
                            (128, 3, 2, 0), 
                            (256, 5, 2, 1), 
                            (256, 5, 3, 1),
                            (128, 5, 3, 1),
                            (64, 3, 2, 1),
                            (64, 3, 2, 1)]
        self.latent_dim_before_quant = latent_dim_before_quant
        self.latent_dim = latent_dim
        # number of levels per dimension in the latent space to be quantized
        self.levels_per_dim = levels_per_dim
        # construct encoder module
        (
            self.encoder_conv_lyrs, 
            self.encoder_fc_mu, 
            self.conv_out_size
        ) = construct_vae_encoder(self.conv_params, self.latent_dim_encoder)
        if self.latent_dim != self.latent_dim_before_quant:
            self.fc_encoder_to_quant = nn.Linear(self.latent_dim_before_quant, self.latent_dim)
        # pass continuous latent vector through discretization bottleneck
        self.vector_quantizer = LatentQuantizer(
                latent_dim = self.latent_dim,
                levels_per_dim = self.levels_per_dim
            )
        print(f"Constructed DLQVAE, with output size after convolution layers: {self.conv_params[-1][0]}X{self.conv_out_size}X{self.conv_out_size}")
        # load trained DLQVAE checkpoint
        checkpoint = torch.load(os.path.join(ENCODERDIR, f'dlqvae_{args.dsName}.pth'))['model']
        # delete the decoder weights as not needed here
        lyr_keys = list(checkpoint.keys())
        for key in lyr_keys:
            if 'decoder' in key or 'encoder_fc_var' in key:
                del checkpoint[key]
        self.load_state_dict(checkpoint)
        print(f"DLQVAE trained for {args.dsName} constructed and loaded successfully!")


    def forward(self, x):
        z = self.encoder_conv_lyrs(x)
        z = torch.flatten(z, start_dim=1)
        z = self.encoder_fc_mu(z)
        if self.latent_dim != self.latent_dim_before_quant:
            z = self.fc_encoder_to_quant(z)
        (
            z_q, 
            quant_idxs, 
            latent_loss_quant, 
            latent_loss_commit
        ) = self.vector_quantizer(z)

        return quant_idxs

    def encode(self, input_data: torch.Tensor) -> torch.Tensor:
        assert input_data.size(1) == 3, "DLQVAE only takes images with 3 channels!"
        with torch.no_grad():
            quantized_indices = self.forward(input_data)
        assert quantized_indices.shape == (input_data.size(0), self.latent_dim)
        assert torch.max(quantized_indices).item() < self.levels_per_dim

        return quantized_indices 
    
    def post_encode(self, encodings):
        return encodings