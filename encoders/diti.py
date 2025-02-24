import os
import sys
import math
import torch
import torch.nn as nn
from torchvision import transforms
from sklearn.cluster import KMeans
from tqdm import trange

sys.path.append("../")
from utils import *

"""
Code adapted from repo https://github.com/yue-zhongqi/diti
"""
LATENT_DIM = 64
LATENT_DIM_RAW = 256
N_DIFFUSION_STEPS = 500
DITI_STAGES = [100,200,300,400,500]    # last number must be the total diffusion steps. E.g., t1,t2 means 2 stages: 0->t1, t1->t2
DITI_DIMS_PER_STAGE = [50,50,50,50,56] 
assert DITI_STAGES[-1] == N_DIFFUSION_STEPS
assert len(DITI_STAGES) == len(DITI_DIMS_PER_STAGE)
assert sum(DITI_DIMS_PER_STAGE) == LATENT_DIM_RAW


def _get_mask_end_dim(idx, t_to_idx):
        if idx >= LATENT_DIM:
            assert False, 'max idx value is k-1!'
        if idx < 0:
            return 0        
        # calculate how many blocks in total after each stage
        accum_num_blocks = np.zeros(len(DITI_STAGES))
        for i in range(LATENT_DIM):
            start_t = torch.nonzero(t_to_idx==i, as_tuple=True)[0][0].item()
            stage = np.argmax(np.array(DITI_STAGES) > start_t)
            accum_num_blocks[stage:] += 1
        start_t = torch.nonzero(t_to_idx==idx, as_tuple=True)[0][0].item()
        stage = np.argmax(np.array(DITI_STAGES) > start_t)
        stage_total_dim = DITI_DIMS_PER_STAGE[stage]
        stage_num_blocks = accum_num_blocks[stage] - accum_num_blocks[stage-1] if stage>0 else accum_num_blocks[stage]
        if int(accum_num_blocks[stage]) == idx + 1:
            return sum(DITI_DIMS_PER_STAGE[0:stage+1])
        else:
            stage_prev_blocks = idx - int(accum_num_blocks[stage-1]) if stage>0 else idx
            return sum(DITI_DIMS_PER_STAGE[0:stage]) + int(float(stage_total_dim) / float(stage_num_blocks)) * (stage_prev_blocks+1)

def construct_latent_groups():
    t_to_idx = torch.zeros(N_DIFFUSION_STEPS).long()        
    for t in range(N_DIFFUSION_STEPS):
        t_to_idx[t] = int(float(t) / (float(N_DIFFUSION_STEPS) / 64))
    latent_groups = torch.zeros(LATENT_DIM, LATENT_DIM_RAW)
    for i in range(LATENT_DIM):
        current_dim = _get_mask_end_dim(i, t_to_idx)
        prev_dim = _get_mask_end_dim(i-1, t_to_idx)
        latent_groups[i, prev_dim:current_dim] = 1.0
    return latent_groups

def normalization(channels):
    """
    Make a standard normalization layer.
    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return nn.GroupNorm(32, channels)

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

class View(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
    def forward(self, input):
        return input.view(self.size)
    
class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )
        weight = torch.softmax(weight, dim=-1)
        a = torch.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )
        weight = torch.softmax(weight, dim=-1)
        a = torch.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)
    
class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """
    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)

class DiTiCELEBAEncoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.latent_dim = kwargs["latent_dim"]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), (2, 2), 1),          # batch_size x 64 x 64 x 64
            normalization(64),
            nn.SiLU(True),
            nn.Conv2d(64, 128, (3, 3), (2, 2), 1),          # batch_size x 128 x 32 x 32
            normalization(128),
            nn.SiLU(True),
            nn.Conv2d(128, 256, (3, 3), (2, 2), 1),         # batch_size x 256 x 16 x 16
            AttentionBlock(256, 4, -1, False),
            normalization(256),
            nn.SiLU(True),
            nn.Conv2d(256, 256, (3, 3), (2, 2), 1),          # batch_size x 256 x 8 x 8
            normalization(256),
            nn.SiLU(True),
            nn.Conv2d(256, 256, (3, 3), (2, 2), 1),          # batch_size x 256 x 4 x 4
            normalization(256),
            nn.SiLU(True),
            View((-1, 256 * 4 * 4)),                  # batch_size x 4096
            nn.Linear(4096, self.latent_dim)
        )

    # x: batch_size x 3 x 128 x 128
    def forward(self, x):
        # batch_size x latent_dim
        return self.encoder(x)

 
class DiTi(nn.Module):
    def __init__(self, 
                 levels_per_dim,
                 args):
        super(DiTi, self).__init__()
        self.latent_dim = LATENT_DIM # k in the original DiTi model
        self.latent_dim_raw = LATENT_DIM_RAW
        self._levels_per_dim = levels_per_dim
        self.encoder = DiTiCELEBAEncoder(latent_dim=self.latent_dim_raw) # length of the latent space to be partitioned into k subsets
        if args.dsName.startswith("mpi3d"):
            dsName_base = "mpi3d"
        elif args.dsName.startswith("celeba"):
            dsName_base = "celeba"
        else:
            dsName_base = args.dsName
        state_dict = torch.load(os.path.join(ENCODERDIR, f'diti_{dsName_base}.pt'))['encoder']
        # With strict=False, other components (diffusion model) are not loaded 
        # for the condition generator
        self.encoder.load_state_dict(state_dict, strict=False)
        self.encoder.to(DEVICE)
        self.encoder.eval()
        self.encoder.requires_grad_(False)

        self.img_size = args.imgSizeToEncoder

        # Follow the same preprocessing transformations in DiTi repo
        self.preprocess_transform = transforms.Normalize((0.5,), (0.5,), inplace=True)

        # for post processing into discrete codes
        self.kmeans = KMeans(n_clusters=self._levels_per_dim, 
                             init='k-means++', 
                             n_init=1, 
                             max_iter=100)

    def encode(self, input_data: torch.Tensor) -> torch.Tensor:
        batch_size = input_data.shape[0]
        assert input_data.shape == (batch_size, 3, self.img_size, self.img_size), \
                    f"Incorrect input image shape for Diti: {input_data.shape}"
        with torch.no_grad():
            input_data = self.preprocess_transform(input_data)
            encodings_outputs_raw = self.encoder.forward(input_data)
        assert encodings_outputs_raw.shape == (batch_size, self.latent_dim_raw)
        return encodings_outputs_raw

    def post_encode(self, encodings_raw):
        print("Diti start post encode...")
        dataset_size = encodings_raw.shape[0]
        assert encodings_raw.shape == (dataset_size, self.latent_dim_raw)
        latent_groups = construct_latent_groups()
        assert latent_groups.sum().item() == LATENT_DIM_RAW
        print("Number of dimensions per latent group: ", latent_groups.sum(1))
        encodings_quantized = []
        for i in trange(self.latent_dim):
            latent_group_mask = latent_groups[i] > 0
            encodings_one_group = encodings_raw[:, latent_group_mask]                        
            encodings_quantized.append(self.kmeans.fit_predict(encodings_one_group))
        encodings_quantized = np.stack(encodings_quantized, axis=1)
        encodings_quantized = torch.from_numpy(encodings_quantized) 
        assert encodings_quantized.shape == (dataset_size, self.latent_dim)
        print("DiTi post_encode computed successfully!")
        return encodings_quantized
