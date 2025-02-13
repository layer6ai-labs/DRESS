import os
import sys
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F 
from safetensors.torch import load_file
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from tqdm import trange
from einops import rearrange, repeat
from diffusers.models import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.unets.unet_2d import UNet2DModel, UNet2DOutput


sys.path.append("../")
from utils import *

"""
Code adapted from repo https://github.com/JindongJiang/latent-slot-diffusion
"""
LATENT_DIM = 14 # the number of attention slots in LSD model
DIM_PER_SLOT = 192 # the length of vector for each slot in LSD model
ATTN_MAP_CLUSTERS = 32

class CartesianPositionalEmbedding(nn.Module):
    def __init__(self, channels, image_size):
        super().__init__()
        self.projection = nn.Conv2d(4, channels, 1)
        self.register_buffer('pe', self.build_grid(image_size).unsqueeze(0))

    def build_grid(self, side_length):
        coords = torch.linspace(0., 1., side_length + 1)
        coords = 0.5 * (coords[:-1] + coords[1:])
        grid_y, grid_x = torch.meshgrid(coords, coords, indexing='xy')
        return torch.stack((grid_x, grid_y, 1 - grid_x, 1 - grid_y), dim=0)

    def forward(self, inputs):
        # `inputs` has shape: [batch_size, out_channels, height, width].
        # `grid` has shape: [batch_size, in_channels, height, width].
        return inputs + self.projection(self.pe)

class UNetEncoder(UNet2DModel):
    def __init__(
        self,
        sample_size: Optional[Union[int, Tuple[int, int]]] = None,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str] = ("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
        up_block_types: Tuple[str] = ("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
        block_out_channels: Tuple[int] = (224, 448, 672, 896),
        layers_per_block: int = 2,
        attention_head_dim: Optional[int] = 8,
        norm_num_groups: int = 32,
        input_resolution: int = 256,
        input_channels: int = 3,
        **kwargs
    ):
        super().__init__(
            sample_size=sample_size,
            in_channels=in_channels,
            out_channels=out_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            attention_head_dim=attention_head_dim,
            norm_num_groups=norm_num_groups,
            **kwargs
        )

        self.register_to_config(
            input_resolution=input_resolution,
            input_channels=input_channels,
        )
        downscale_stride = input_resolution // sample_size
        self.downscale_cnn = nn.Conv2d(input_channels, in_channels, kernel_size=downscale_stride, stride=downscale_stride)
        self.original_forward = super().forward

    def forward(
        self,
        sample: torch.FloatTensor,
    ) -> Union[UNet2DOutput, Tuple]:

        sample = self.downscale_cnn(sample)
        return self.original_forward(sample, timestep=0, class_labels=None).sample
    
class MultiHeadSTEVESA(ModelMixin, ConfigMixin):
    # enable diffusers style config and model save/load
    @register_to_config
    def __init__(self, num_iterations, num_slots, num_heads,
                 input_size, out_size, slot_size, mlp_hidden_size, 
                 input_resolution, epsilon=1e-8, 
                 learnable_slot_init=False, 
                 bi_level=False):
        super().__init__()

        self.pos = CartesianPositionalEmbedding(input_size, input_resolution)
        self.in_layer_norm = nn.LayerNorm(input_size)
        self.in_mlp = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.ReLU(),
            nn.Linear(input_size, input_size)
            )
        if bi_level:
            # We tested bi_level slot attention (Jia et al. in https://arxiv.org/abs/2210.08990) at the early stage of the project,
            # and we didn't find it helpful
            assert learnable_slot_init, 'Bi-level training requires learnable_slot_init=True'

        self.num_iterations = num_iterations
        self.num_slots = num_slots
        self.num_heads = num_heads
        self.input_size = input_size
        self.slot_size = slot_size
        self.mlp_hidden_size = mlp_hidden_size
        self.epsilon = epsilon
        self.learnable_slot_init = learnable_slot_init
        self.bi_level = bi_level

        assert slot_size % num_heads == 0, 'slot_size must be divisible by num_heads'

        if learnable_slot_init:
            self.slot_mu = nn.Parameter(torch.Tensor(1, num_slots, slot_size))
            nn.init.xavier_uniform_(self.slot_mu)
        else:
            # parameters for Gaussian initialization (shared by all slots).
            self.slot_mu = nn.Parameter(torch.Tensor(1, 1, slot_size))
            self.slot_log_sigma = nn.Parameter(torch.Tensor(1, 1, slot_size))
            nn.init.xavier_uniform_(self.slot_mu)
            nn.init.xavier_uniform_(self.slot_log_sigma)

        # norms
        self.norm_inputs = nn.LayerNorm(input_size)
        self.norm_slots = nn.LayerNorm(slot_size)
        self.norm_mlp = nn.LayerNorm(slot_size)

        # linear maps for the attention module.
        self.project_q = nn.Linear(slot_size, slot_size, bias=False)
        self.project_k = nn.Linear(input_size, slot_size, bias=False)
        self.project_v = nn.Linear(input_size, slot_size, bias=False)

        # slot update functions.
        self.gru = nn.GRUCell(slot_size, slot_size)
        self.mlp = nn.Sequential(
            nn.Linear(slot_size, mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(mlp_hidden_size, slot_size))
        
        self.out_layer_norm = nn.LayerNorm(slot_size)
        self.out_linear = nn.Linear(slot_size, out_size)
        
    def forward(self, inputs):
        slots_collect, attns_collect = self.forward_slots(inputs)
        slots_collect = self.out_layer_norm(slots_collect)
        slots_collect = self.out_linear(slots_collect)
        return slots_collect, attns_collect

    def forward_slots(self, inputs):
        """
        inputs: batch_size x seq_len x input_size x h x w
        return: batch_size x num_slots x slot_size
        """
        B, T, input_size, h, w = inputs.size()
        inputs = self.pos(inputs)
        inputs = rearrange(inputs, 'b t n_inp h w -> b t (h w) n_inp')
        inputs = self.in_mlp(self.in_layer_norm(inputs))

        if self.learnable_slot_init:
            slots = repeat(self.slot_mu, '1 num_s d -> b num_s d', b=B)
        else:
            # initialize slots
            slots = inputs.new_empty(B, self.num_slots, self.slot_size).normal_()
            slots = self.slot_mu + torch.exp(self.slot_log_sigma) * slots

        # setup key and value
        inputs = self.norm_inputs(inputs)
        k = rearrange(self.project_k(inputs), 'b t n_inp (h d) -> b t h n_inp d',
                      h=self.num_heads)  # Shape: [batch_size, T, num_heads, num_inputs, slot_size].
        v = rearrange(self.project_v(inputs), 'b t n_inp (h d) -> b t h n_inp d',
                      h=self.num_heads)  # Shape: [batch_size, T, num_heads, num_inputs, slot_size].
        k = (self.slot_size ** (-0.5)) * k

        # loop over frames
        attns_collect = []
        slots_collect = []
        for t in range(T):
            # corrector iterations
            for i in range(self.num_iterations):
                if self.bi_level and i == self.num_iterations - 1:
                    slots = slots.detach() + self.slot_mu - self.slot_mu.detach()
                slots_prev = slots
                slots = self.norm_slots(slots)

                # Attention.
                q = rearrange(self.project_q(slots), 'b n_s (h d) -> b h n_s d',
                              h=self.num_heads)  # Shape: [batch_size, num_heads, num_slots, slot_size].
                attn_logits = torch.einsum('...id,...sd->...is', k[:, t],
                                           q)  # Shape: [batch_size, num_heads, num_inputs, num_slots]
                attn = F.softmax(rearrange(attn_logits, 'b h n_inp n_s -> b n_inp (h n_s)'), -1)
                attn_vis = rearrange(attn, 'b n_inp (h n_s) -> b h n_inp n_s', h=self.num_heads)
                # `attn_vis` has shape: [batch_size, num_inputs, num_slots].

                # Weighted mean.
                attn = attn_vis + self.epsilon
                attn = attn / torch.sum(attn, dim=-2, keepdim=True)  # norm over inputs
                updates = torch.einsum('...is,...id->...sd', attn,
                                       v[:, t])  # Shape: [batch_size, num_heads, num_slots, num_inp].
                updates = rearrange(updates, 'b h n_s d -> b n_s (h d)')
                # `updates` has shape: [batch_size, num_slots, slot_size].

                # Slot update.
                slots = self.gru(updates.view(-1, self.slot_size),
                                 slots_prev.reshape(-1, self.slot_size))
                slots = slots.view(-1, self.num_slots, self.slot_size)
                slots = slots + self.mlp(self.norm_mlp(slots))

            # collect
            attns_collect += [attn_vis]
            slots_collect += [slots]

        attns_collect = torch.stack(attns_collect, dim=1)  # B, T, num_inputs, num_slots
        slots_collect = torch.stack(slots_collect, dim=1)  # B, T, num_slots, slot_size

        return slots_collect, attns_collect

class LSD(nn.Module):
    def __init__(self, 
                 levels_per_dim,
                 args):
        super(LSD, self).__init__()
        self.latent_dim = LATENT_DIM # the number of attention slots in LSD model
        self.dim_per_slot = DIM_PER_SLOT
        self._levels_per_dim = levels_per_dim
        backbone_config = UNetEncoder.load_config(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "lsd_backbone_config.json"))
        self.img_to_latent_encoder = UNetEncoder.from_config(backbone_config)
        slot_attn_config = MultiHeadSTEVESA.load_config(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "lsd_slta_config.json"))
        self.slot_attention_encoder = MultiHeadSTEVESA.from_config(slot_attn_config)
         
        if args.dsName.startswith("mpi3d"):
            dsName_base = "mpi3d"
        elif args.dsName.startswith("celeba"):
            dsName_base = "celeba"
        else:
            dsName_base = args.dsName

        self.img_to_latent_encoder.load_state_dict(
            load_file(os.path.join(ENCODERDIR, f'lsd_{dsName_base}_backbone_{self.latent_dim}D_latent.safetensors')), strict=True)
        self.img_to_latent_encoder.to(DEVICE)
        self.img_to_latent_encoder.eval()
        self.img_to_latent_encoder.requires_grad_(False)
        self.slot_attention_encoder.load_state_dict(
            load_file(os.path.join(ENCODERDIR, f'lsd_{dsName_base}_slta_{self.latent_dim}D_latent.safetensors')), strict=True)
        self.slot_attention_encoder.to(DEVICE)
        self.slot_attention_encoder.eval()
        self.slot_attention_encoder.requires_grad_(False)

        self.img_size = args.imgSizeToEncoder

        # for clustering attention maps
        self.attn_kmeans_model = KMeans(n_clusters=ATTN_MAP_CLUSTERS,
                                  init="k-means++",
                                  n_init=1,
                                  max_iter=100)
        self.attn_kmeans_fitted = False
        # for post processing into discrete codes
        self.postencode_kmeans_model = KMeans(n_clusters=self._levels_per_dim, 
                                        init='k-means++', 
                                        n_init=1, 
                                        max_iter=100)
        
        print(f"Created LSD model with {self.latent_dim} latent dimensions!")

    def encode(self, input_data: torch.Tensor) -> torch.Tensor:
        batch_size = input_data.shape[0]
        assert input_data.shape == (batch_size, 3, self.img_size, self.img_size), \
                    f"Incorrect input image shape for LSD: {input_data.shape}"
        with torch.no_grad():
            intermediate_latents = self.img_to_latent_encoder(input_data)
            slots, attn = self.slot_attention_encoder(intermediate_latents[:,None])
        assert slots.shape == (batch_size, 1, self.latent_dim, self.dim_per_slot)
        assert attn.shape == (batch_size, 1, 1, 4096, self.latent_dim)
        slots = slots[:, 0]
        attn = attn[:, 0, 0]
        attn = rearrange(attn, 'b l s -> b s l')
        # normalize attn so the 4096-dim map lives in a unit sphere
        attn = attn / attn.sum(dim=2, keepdim=True)
        if not self.attn_kmeans_fitted:
            self.attn_kmeans_model = self.attn_kmeans_model.fit(attn.reshape(-1, 4096).cpu())
            self.attn_kmeans_fitted = True
        slots_cluster_ids = np.reshape(self.attn_kmeans_model.predict(attn.reshape(-1, 4096).cpu()), 
                                 [batch_size, self.latent_dim])    
        slots_order = torch.tensor(np.argsort(slots_cluster_ids, axis=1))
        slots_order = slots_order.unsqueeze(2).repeat(1,1,self.dim_per_slot).to(DEVICE)
        slots = torch.gather(slots, dim=1, index=slots_order)
        return slots

    def post_encode(self, encodings_raw):
        print("LSD start post encode...")
        dataset_size = encodings_raw.shape[0]
        assert encodings_raw.shape == (dataset_size, self.latent_dim, self.dim_per_slot)
        encodings_quantized = [self.postencode_kmeans_model.fit_predict(encodings_raw[:,i])
                                for i in trange(self.latent_dim)]
        encodings_quantized = np.stack(encodings_quantized, axis=1)
        encodings_quantized = torch.from_numpy(encodings_quantized) 
        assert encodings_quantized.shape == (dataset_size, self.latent_dim)
        print("LSD post_encode computed successfully!")
        return encodings_quantized
