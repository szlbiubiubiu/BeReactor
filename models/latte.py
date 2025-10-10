# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------
import math
import torch
import torch.nn as nn
import numpy as np

from einops import rearrange, repeat
from timm.models.vision_transformer import Mlp, PatchEmbed
import torch.nn.functional as Func
import torch.nn.functional as F
from timm.layers import  DropPath

import pdb
from models.emo_fr_net.model_irse import IDnet
from models.emo_fr_net.basenet import LNDnet
from models.emo_fr_net.seg_model_2 import BGnet
from models.emo_fr_net.emo_net import EmoNet

from models.emo_fr_net.au_model import MEFL
# for i in sys.path:
#     print(i)

# the xformers lib allows less memory, faster training and inference
from face_alignment.detection.sfd.sfd_detector import SFDDetector
from math import sqrt

try:
    import xformers
    import xformers.ops
except:
    XFORMERS_IS_AVAILBLE = False

# from timm.models.layers.helpers import to_2tuple
# from timm.models.layers.trace_utils import _assert

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

#################################################################################
#               Attention Layers from TIMM                                      #
#################################################################################

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., use_lora=False, attention_mode='math'):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.attention_mode = attention_mode
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        
        if self.attention_mode == 'xformers': # cause loss nan while using with amp
            x = xformers.ops.memory_efficient_attention(q, k, v).reshape(B, N, C)

        elif self.attention_mode == 'flash':
            # cause loss nan while using with amp
            # Optionally use the context manager to ensure one of the fused kerenels is run
            with torch.backends.cuda.sdp_kernel(enable_math=False):
                x = torch.nn.functional.scaled_dot_product_attention(q, k, v).reshape(B, N, C) # require pytorch 2.0

        elif self.attention_mode == 'math':
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        else:
            raise NotImplemented

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t, use_fp16=False):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        if use_fp16:
            t_freq = t_freq.to(dtype=torch.float16)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings

class image_drop_emb(nn.Module):
    def __init__(self,  dropout_prob=0.0):
        super().__init__()
        self.dropout_prob=dropout_prob

    def forward(self,image,training=False):
        use_dropout = self.dropout_prob > 0
        if (training and use_dropout):
            # pdb.set_trace()
            drop_ids = torch.rand((image.shape[0],1,1,1,1), device=image.device) < self.dropout_prob
            image = torch.where(drop_ids, torch.zeros_like(image), image)
            # zero_image=torch.zeros_like(image)
            # image=zero_image
        return image

class feature_drop_emb(nn.Module):
    def __init__(self,  dropout_prob=0.0):
        super().__init__()
        self.dropout_prob=dropout_prob

    def forward(self,image,training=False):
        use_dropout = self.dropout_prob > 0
        if (training and use_dropout):
            # pdb.set_trace()
            drop_ids = torch.rand((image.shape[0],1), device=image.device) < self.dropout_prob
            image = torch.where(drop_ids, torch.zeros_like(image), image)
            # zero_image=torch.zeros_like(image)
            # image=zero_image
        return image
#################################################################################
#                                 Core Latte Model                                #
#################################################################################

class TransformerBlock(nn.Module):
    """
    A Latte tansformer block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        # pdb.set_trace()
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of Latte.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class Latte(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        num_frames=16,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
        extras=3,
        attention_mode='math',
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.extras = extras
        self.num_frames = num_frames
        
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)

        if self.extras == 2:
            self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
            
        elif self.extras==3 :# facial condition

            # self.facial_condition = DiffusionVideoAutoencoder("./pretrained_models/model_ir_se50.pth", "./pretrained_models/mobilenet_224_model_best_gdconv_external.pth.tar", "./pretrained_models/79999_iter.pth"
            #         ,emotion_file_path='./pretrained_models/emonet_8.pth', au_file_path_disfa='./pretrained_models/MEFARG_swin_base_DISFA_fold1.pth',au_file_path_bp4d='./pretrained_models/MEFARG_swin_base_BP4D_fold1.pth')
            # self.facial_condition = DiffusionVideoAutoencoder_v2("./pretrained_models/model_ir_se50.pth", "./pretrained_models/mobilenet_224_model_best_gdconv_external.pth.tar", "./pretrained_models/79999_iter.pth"
            #         ,emotion_file_path='./pretrained_models/emonet_8.pth', au_file_path_disfa='./pretrained_models/MEFARG_swin_base_DISFA_fold1.pth',au_file_path_bp4d='./pretrained_models/MEFARG_swin_base_BP4D_fold1.pth')
            self.facial_condition = DiffusionVideoAutoencoder_v3("./pretrained_models/model_ir_se50.pth", "./pretrained_models/mobilenet_224_model_best_gdconv_external.pth.tar", "./pretrained_models/79999_iter.pth"
                    ,emotion_file_path='./pretrained_models/emonet_8.pth', au_file_path_disfa='./pretrained_models/MEFARG_swin_base_DISFA_fold1.pth',au_file_path_bp4d='./pretrained_models/MEFARG_swin_base_BP4D_fold1.pth')
            
            
            # self.sp_drop_operation=image_drop_emb(0.1)
            # self.lis_drop_operation=image_drop_emb(0.1)
            self.sp_drop_operation=image_drop_emb(0.2)
            self.lis_drop_operation=image_drop_emb(0.2)
            self.feat_drop_operation=feature_drop_emb(0.1)
        if self.extras == 78: # timestep + text_embedding
            self.text_embedding_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(77 * 768, hidden_size, bias=True)
        )

        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        self.temp_embed = nn.Parameter(torch.zeros(1, num_frames, hidden_size), requires_grad=False)
        self.hidden_size =  hidden_size

        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, attention_mode=attention_mode) for _ in range(depth)
        ])

        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        temp_embed = get_1d_sincos_temp_embed(self.temp_embed.shape[-1], self.temp_embed.shape[-2])
        self.temp_embed.data.copy_(torch.from_numpy(temp_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        if self.extras == 2:
            # Initialize label embedding table:
            nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in Latte blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    # @torch.cuda.amp.autocast()
    # @torch.compile
    def forward(self, 
                x, 
                t, 
                y=None, 
                original_x=None,
                listener_ref=None,
                listener_video_clip_last=None,
                sp_audio_clip=None,
                text_embedding=None, 
                use_fp16=False):
        """
        Forward pass of Latte.
        x: (N, F, C, H, W) tensor of video inputs
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        if use_fp16:
            x = x.to(dtype=torch.float16)

        batches, frames, channels, high, weight = x.shape 
        x = rearrange(x, 'b f c h w -> (b f) c h w')
        # pdb.set_trace()
        x = self.x_embedder(x) + self.pos_embed  
        t = self.t_embedder(t, use_fp16=use_fp16)                  
        timestep_spatial = repeat(t, 'n d -> (n c) d', c=self.temp_embed.shape[1]) 
        timestep_temp = repeat(t, 'n d -> (n c) d', c=self.pos_embed.shape[1])

        if self.extras == 2:
            y = self.y_embedder(y, self.training)
            y_spatial = repeat(y, 'n d -> (n c) d', c=self.temp_embed.shape[1]) 
            y_temp = repeat(y, 'n d -> (n c) d', c=self.pos_embed.shape[1])
        elif self.extras == 3:
            # 
            if self.training is True:
                # pdb.set_trace()
                original_x=self.sp_drop_operation(original_x,training=self.training)
                listener_ref=self.lis_drop_operation(listener_ref,training=self.training)
            facial_cond,hidden_stats,id_feats,listener_id_feats=self.facial_condition(original_x,listener_ref=listener_ref,
                    listener_video_clip_last=listener_video_clip_last,speaker_Audio=sp_audio_clip)
            # facial_cond=self.feat_drop_operation(facial_cond,training=self.training)
            facial_spatial = repeat(facial_cond, 'n d -> (n c) d', c=self.temp_embed.shape[1]) 
            facial_temp = repeat(facial_cond, 'n d -> (n c) d', c=self.pos_embed.shape[1])
        elif self.extras == 78:
            text_embedding = self.text_embedding_projection(text_embedding.reshape(batches, -1))
            text_embedding_spatial = repeat(text_embedding, 'n d -> (n c) d', c=self.temp_embed.shape[1])
            text_embedding_temp = repeat(text_embedding, 'n d -> (n c) d', c=self.pos_embed.shape[1])
        # pdb.set_trace()
        for i in range(0, len(self.blocks), 2):
            spatial_block, temp_block = self.blocks[i:i+2]
            if self.extras == 2:
                c = timestep_spatial + y_spatial
            elif self.extras == 3:
                c = timestep_spatial + facial_spatial
            elif self.extras == 78:
                c = timestep_spatial + text_embedding_spatial
            else:
                c = timestep_spatial
            # pdb.set_trace()
            x  = spatial_block(x, c)

            x = rearrange(x, '(b f) t d -> (b t) f d', b=batches)
            # Add Time Embedding
            if i == 0:
                x = x + self.temp_embed

            if self.extras == 2:
                c = timestep_temp + y_temp
            elif self.extras == 3:
                c= timestep_temp + facial_temp
            elif self.extras == 78:
                c = timestep_temp + text_embedding_temp
            else:
                c = timestep_temp

            x = temp_block(x, c)
            x = rearrange(x, '(b t) f d -> (b f) t d', b=batches)
        # pdb.set_trace()
        if self.extras == 2:
            c = timestep_spatial + y_spatial
        elif self.extras == 3:
            c = timestep_spatial + facial_spatial
        else:
            c = timestep_spatial
        x = self.final_layer(x, c)    
        # pdb.set_trace()           
        x = self.unpatchify(x)                  
        x = rearrange(x, '(b f) c h w -> b f c h w', b=batches)
        return x
    def forward_with_cfg_reaction(self, x, t,original_x=None,listener_ref=None, y=None, cfg_scale=7.0, use_fp16=False, text_embedding=None):
        """
        Forward pass of Latte, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        # pdb.set_trace()
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)

        #for condition, copy and set to zero
        half_x = original_x[: len(x) // 2]
        combined_originalx = torch.cat([half_x, half_x], dim=0)

        half_listener_ref = listener_ref[: len(x) // 2]
        combined_listener_ref = torch.cat([half_listener_ref, torch.zeros_like(half_listener_ref)], dim=0)
        if use_fp16:
            combined = combined.to(dtype=torch.float16)
        model_out = self.forward(combined, t,original_x=combined_originalx,listener_ref=combined_listener_ref, y=y, use_fp16=use_fp16, text_embedding=text_embedding)
        # pdb.set_trace()
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        # eps, rest = model_out[:, :3], model_out[:, 3:]
        eps, rest = model_out[:, :, :4, ...], model_out[:, :, 4:, ...] 
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0) 
        # pdb.set_trace()
        return torch.cat([eps, rest], dim=2)
    def forward_with_cfg(self, x, t, y=None, cfg_scale=7.0, use_fp16=False, text_embedding=None):
        """
        Forward pass of Latte, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        if use_fp16:
            combined = combined.to(dtype=torch.float16)
        model_out = self.forward(combined, t, y=y, use_fp16=use_fp16, text_embedding=text_embedding)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        # eps, rest = model_out[:, :3], model_out[:, 3:]
        eps, rest = model_out[:, :, :4, ...], model_out[:, :, 4:, ...] 
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0) 
        return torch.cat([eps, rest], dim=2)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
class ConvBlock(nn.Module):
    def __init__(self, in_planes=3, planes=128):
        super(ConvBlock, self).__init__()
        self.planes = planes
        self.conv1 = nn.Conv3d(in_planes, planes // 4, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3),
                               bias=False)
        self.in1 = nn.InstanceNorm3d(planes // 4)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 0, 0))

        self.conv2 = nn.Conv3d(planes // 4, planes, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3),
                               bias=False)
        self.in2 = nn.InstanceNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)

        # self.conv3 = nn.Conv3d(planes, planes, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), bias=False)
        # self.in3 = nn.InstanceNorm3d(planes)
        # self.relu = nn.ReLU(inplace=True)

        # self.conv4 = nn.Conv3d(planes, planes, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False)
        # self.in4 = nn.InstanceNorm3d(planes)
        # self.relu = nn.ReLU(inplace=True)


        # self.conv5 = nn.Conv3d(planes, planes, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), bias=False)
        # self.in5 = nn.InstanceNorm3d(planes)
        # self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        input:
        speaker_video_frames x: (batch_size, 3, seq_len, img_size, img_size)

        output:
        speaker_temporal_tokens y: (batch_size, token_dim, seq_len)

        """

        x = self.relu(self.in1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.relu(self.in2(self.conv2(x)))
        # x = self.relu(self.in3(self.conv3(x)))
        # x = x + self.relu(self.in4(self.conv4(x)))
        # x = self.relu(self.in5(self.conv5(x)))

        y = x.mean(dim=-1).mean(dim=-1).mean(dim=-1)
        return y
class DiffusionVideoAutoencoder(nn.Module):
    def __init__(self, id_file_path, lnd_file_path, bg_file_path,au_file_path_disfa=None,
    au_file_path_bp4d=None,emotion_file_path=None, detect_face=True):
        super(DiffusionVideoAutoencoder, self).__init__()
        self.idnet = IDnet(id_file_path).requires_grad_(False)#don't have to modify resolution
        # self.idnet.eval()
        # self.lndnet = LNDnet(lnd_file_path).requires_grad_(False)#modify resolution
        self.bgnet = BGnet(bg_file_path).requires_grad_(False)#don't have to modify resolution
        self.linear = nn.Linear(512, 384)#model s:384, model B: 768 #With listener 512
        # self.lstm = nn.LSTM(512, 512, 1 , batch_first = True)
        self.emo_proj=nn.Sequential(nn.Linear(25,128),
                                    nn.Dropout(p=0.5)
                                    )
        self.id_proj=nn.Sequential(nn.Linear(512,64),#64-128
                                    # nn.Dropout(p=0.5)
                                    )
        self.emo_cond_linear=nn.Linear(512*2,512)
        self.encode_video=ConvBlock()
        self.encode_video_listener=ConvBlock()
        # self.sp_drop=DropPath(0.5)
        # self.lis_drop=DropPath(0.5)
        self.sp_drop=feature_drop_emb(0.2)
        self.lis_drop=feature_drop_emb(0.2)
        if au_file_path_disfa is not None:
            # pdb.set_trace()
            self.au_net_DISFA=MEFL.MEFARG(num_classes=8,backbone='resnet18').requires_grad_(False)# resnet
            # self.au_net_DISFA.load_state_dict(torch.load(au_file_path_disfa, map_location='cuda')['state_dict'])
            # self.au_net_DISFA.eval()
            self.au_net_BP4D=MEFL.MEFARG(num_classes=12,backbone='resnet18').requires_grad_(False) #num_classes: 12   neighbor_num: 4
            # self.au_net_BP4D.load_state_dict(torch.load(au_file_path_bp4d, map_location='cuda')['state_dict'])
            # self.au_net_BP4D.eval()
        if emotion_file_path is not None:
            #input resolution is 256
            
            self.emo_net=EmoNet(n_expression=8).requires_grad_(False)#what is the resolution#.to(device)()
            
            state_dict = torch.load(emotion_file_path, map_location="cpu")
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            self.emo_net.load_state_dict(state_dict,strict=True)
            # self.emo_net.load_state_dict(torch.load(emotion_file_path, map_location='cuda'),strict=True)
            
            self.emo_net.eval()
        if detect_face is not None:
            device = "cuda:0"
            self.sfd_detector = SFDDetector(device)
        self.resolution = torch.nn.AdaptiveAvgPool2d((256, 256))
    def forward(self, id,listener_ref=None, listener_video_clip_last=None, lnd=None, emo_cond=None, return_id_feats=False,
                return_emo=False,expression=False):
        # pdb.set_trace()
        # if emo_cond is not None:
        #     listener_ref=emo_cond['listener_ref']
        # else:
        #     listener_ref=None
        # hidden_stats=emo_cond['hidden_stats']
        # past_emo=None#emo_cond['past_emo']
        # pdb.set_trace()
        # if len(id.shape)==4:
        #     if lnd is None:
        #         lnd = id

        #     id_feats = self.idnet(id)  # [B, 512]
        #     # lnd_feats = self.lndnet(lnd) # [B, 102]
        #     feats = self.linear(torch.cat([id_feats, lnd_feats], dim=1))   # [B, 512]
            
        #     if return_id_feats:
        #         return feats, id_feats
        #     return feats
        # else:  #video process
            # pdb.set_trace()
            
        if lnd is None:
            lnd = id
        # pdb.set_trace()
        b,v,c,w,h=id.shape
        b,v,c_l,w_l,h_l=lnd.shape
        # pdb.set_trace()
        
        video_input=id.permute(0,2,1,3,4)
        encode_fea=self.encode_video(video_input) #[b,128]
        # pdb.set_trace()
        
        id_feats,emo_fea,au_disfa,au_bp4d=self.forward_listener(id)
        id=id.reshape(b*v,c,w,h)
        lnd=lnd.reshape(b*v,c_l,w_l,h_l)
        
        # # pdb.set_trace()
        # self.idnet.eval()
        # id_feats = self.idnet(id)  # [B, 512]

        #emo_feature
        # pdb.set_trace()
        au=torch.cat([au_bp4d, au_disfa[:, 4:5], au_disfa[:,-2:]],dim=-1)  #[8*b,15]->[b,8,15]
        au=torch.reshape(au,[b,-1,15])

        #
        expression=emo_fea['expression']
        expression_soft=Func.softmax(expression,dim=1)
        valence=emo_fea['valence']
        arousal=emo_fea['arousal']
        emo=torch.cat([expression_soft,torch.unsqueeze(valence,dim=-1),torch.unsqueeze(arousal,dim=-1)],dim=-1)
        emo=torch.reshape(emo,[b,-1,10])
        current_emo=torch.cat([au,emo],dim=-1)


       
        lisenter_ref_id= self.idnet(listener_ref[:,0,:,:,:])
        # lisenter_ref_id=self.lis_drop(lisenter_ref_id,training=self.training)
        # lnd_feats = self.lndnet(lnd) # [B, 102]
        # pdb.set_trace()
        # lis_drop
        # print(listener_video_clip_last.shape)
        if listener_video_clip_last is not None:
            video_input_listner_last=listener_video_clip_last.permute(0,2,1,3,4)
            lisenter_encode_fea=self.encode_video_listener(video_input_listner_last)
            # print(encode_fea.shape)
            # print(lisenter_encode_fea.shape)
            feats = self.linear(torch.cat([self.id_proj(id_feats), self.emo_proj(current_emo).mean(-2),
                encode_fea,lisenter_encode_fea,self.id_proj(lisenter_ref_id)], dim=1))   # [B, 64+128+128+128+64=512] #
        else:
            feats = self.linear(torch.cat([self.id_proj(id_feats), self.emo_proj(current_emo).mean(-2),
                encode_fea,self.id_proj(lisenter_ref_id)], dim=1))   # [B, 64+128+128+64=384]

        # feats = self.linear(torch.cat([self.emo_proj(current_emo).mean(-2),
        #         encode_fea,self.id_proj(lisenter_ref_id)], dim=1))   # [B, 512]
        # feats = feats.reshape(b,v,-1)
        # id_feats=id_feats.reshape(b,v,-1)


        

        # if len(feats.shape)==3:
        #     feats=torch.nn.functional.avg_pool2d(feats, (feats.shape[1], 1), stride=(feats.shape[1], 1)).squeeze()
        #     if id_feats is not None:
        #         id_feats = torch.nn.functional.avg_pool2d(id_feats, (id_feats.shape[1], 1), stride=(id_feats.shape[1], 1))#.squeeze()
        # # if return_emo:
        # #     # pdb.set_trace()
        # #     # self.emo_net.eval()
        # #     emo_fea=self.emo_net(id)
        # #     # emo_fea
        # #     # self.au_net_DISFA.eval()
        # #     # self.au_net_BP4D.eval()
        # #     au_disfa=self.au_net_DISFA(id)[0]
        # #     au_bp4d=self.au_net_BP4D(id)[0]

        # #     return feats,emo_fea,au_disfa,au_bp4d,hidden_stats
        # listener_id_feats = self.idnet(listener_ref[:,0,:,:,:]) 
        hidden_stats=None
        # # if listener_ref != None and past_emo!=None:
        # #     #1 get listener id
        # #     listener_id_feats = self.idnet(listener_ref) 
        # #     # speaker_feat=torch.cat([id_feats,au_disfa,au_bp4d,])
        # #     # pdb.set_trace()
        # #     #2 past emotion with lstm
        # #     past_emo=torch.nn.functional.avg_pool2d(past_emo, (past_emo.shape[1], 1), stride=(past_emo.shape[1], 1))#.squeeze()
        # #     # pdb.set_trace()
        # #     listener_temporal_emo, hidden_stats = self.lstm(self.emo_proj(past_emo),hidden_stats)# (hidden_stats, hidden_stats))
        # #     # feats=self.emo_cond_linear(
        # #     #     torch.cat([listener_id_feats,listener_temporal_emo.squeeze(),feats],dim=-1))#fuse speaker+ listener condition
        # #     # listener_temporal_emo, hidden_stats = self.lstm(self.emo_proj(id_feats),hidden_stats)# (hidden_stats, hidden_stats))
        # #     feats=self.emo_cond_linear(
        # #         torch.cat([listener_temporal_emo.squeeze(),feats],dim=-1))#fuse speaker+ listener condition
        # #     id_feats=id_feats.squeeze()
        # # pdb.set_trace()
        return feats,hidden_stats,id_feats,lisenter_ref_id
        # if return_id_feats:
        #     return feats, id_feats
        # return feats
    def forward_listener(self, generated_listener,emo_cond=None,return_emo=False,detect_face=False):
        # pdb.set_trace()
        id=generated_listener
        if len(generated_listener.shape)==5:
            b,v,c,w,h=id.shape
            # b,v,c_l,w_l,h_l=lnd.shape

            id=id.reshape(b*v,c,w,h)
        elif len(generated_listener.shape)==4:
            b,c,w,h=id.shape
            
        # b,v,c,w,h=id.shape
        # # b,v,c_l,w_l,h_l=lnd.shape

        # id=id.reshape(b*v,c,w,h)
        # lnd=lnd.view(b*v,c_l,w_l,h_l)

        # pdb.set_trace()
        id_feats = self.idnet(id)  # [B, 512]
        # lnd_feats = self.lndnet(id) # [B, 102]
        # fea
        # feats = self.linear(torch.cat([id_feats, lnd_feats], dim=1))   # [B, 512]
        # feats = feats.view(b,v,-1)
        
        if len(generated_listener.shape)==5:
            id_feats=id_feats.reshape(b,v,-1)


            

            # if len(feats.shape)==3:
            #     feats=torch.nn.functional.avg_pool2d(feats, (feats.shape[1], 1), stride=(feats.shape[1], 1)).squeeze()
            if id_feats is not None:
                id_feats = torch.nn.functional.avg_pool2d(id_feats, (id_feats.shape[1], 1), stride=(id_feats.shape[1], 1)).squeeze()
        elif len(generated_listener.shape)==4:
            # pdb.set_trace()
            if id_feats is not None:
                id_feats = id_feats.mean(dim=0)
                # torch.nn.functional.avg_pool2d(id_feats, (id_feats.shape[0], 1), stride=(id_feats.shape[0], 1)).squeeze()
        if detect_face == True:
            # pdb.set_trace()
            detect_face=self.sfd_detector.detect_from_batch(128*(id/2+0.5))
            emo_id=torch.zeros_like(id)
            for index,frame in enumerate(id):
                    
                # detected_faces = self.sfd_detector.detect_from_image(frame)#[:, :, ::-1]) #BGR frame
                
                # for each f
                x_min, y_min, x_max, y_max,confidence = detect_face[index][0]
                
                cropped_face = frame[:,np.int(y_min):np.int(y_max), np.int(x_min):np.int(x_max)]
                # else:
                #     crop_frame = frame
                resized_frame = F.interpolate(cropped_face, size=(256, 256), mode='bilinear', align_corners=False)
                # resized_frame=self.resolution(cropped_face)
                emo_id[index]=resized_frame
                # crop_frame = cv2.resize(cropped_face,(256,256),interpolation = cv2.INTER_LANCZOS4)
                # id[index]=detected_faces

            # pdb.set_trace()
            with torch.no_grad():
                emo_fea=self.emo_net(emo_id)
            pdb.set_trace()
        else:
            emo_fea=self.emo_net(id)
        
        # expression_soft=F.softmax(emo_fea,dim=1)
        # emo_fea
        # self.au_net_DISFA.eval()
        # self.au_net_BP4D.eval()
        au_disfa=self.au_net_DISFA(id)[0]
        au_bp4d=self.au_net_BP4D(id)[0]



        # if listener_ref != None and past_emo!=None:
        #     #1 get listener id
        #     listener_id_feats = self.idnet(listener_ref) 
        #     speaker_feat=torch.cat([id_feats,au_disfa,au_bp4d,])

        #     #2 past emotion with lstm
        #     listener_temporal_emo, hidden_status = self.lstm(self.emo_proj(past_emo[:,-1]), hidden_status, 
        #                             hidden_status)
        #     feats=torch.cat([listener_id_feats,listener_temporal_emo,feats])#fuse speaker+ listener condition
        #     # au_bp4d=model_forward.au_bp4d
        #     # au_disfa=model_forward.au_disfa

        #     # # pdb.set_trace()
        #     # #
        #     # au=torch.cat([au_bp4d, au_disfa[:, 4:5], au_disfa[:,-2:]],dim=-1)  #[8*b,15]->[b,8,15]
        #     # au=torch.reshape(au,[len(generated_listener),-1,15])

        #     # #
        #     # emo=torch.cat([expression_soft,torch.unsqueeze(valence,dim=-1),torch.unsqueeze(arousal,dim=-1)],dim=-1)
        #     # emo=torch.reshape(emo,[len(generated_listener),-1,10])
        #     # #au loss
        #     # au_loss= mean_flat((au-emo_label[:,:,:15])**2)
        #     # emo_loss= mean_flat((emo-emo_label[:,:,15:])**2)
        return id_feats,emo_fea,au_disfa,au_bp4d
        # if return_id_feats:
        #     return feats, id_feats
        # return feats
    
    def id_forward(self, id):
        return self.idnet(id)  # [B, 512]
        
    def forward_with_id(self, id_feats, lnd):
        lnd_feats = self.lndnet(lnd) # [B, 102]
        feats = self.linear(torch.cat([id_feats, lnd_feats], dim=1))   # [B, 512]
        return feats
    # def forward_emo(self,input_img):

    def face_mask(self, bg, for_video=False):
        fg_mask = self.bgnet(bg, for_video=for_video) # [B, 1, 256, 256]
        return fg_mask
class DiffusionVideoAutoencoder_v2(nn.Module):
    def __init__(self, id_file_path, lnd_file_path, bg_file_path,au_file_path_disfa=None,
    au_file_path_bp4d=None,emotion_file_path=None, detect_face=True):
        super(DiffusionVideoAutoencoder_v2, self).__init__()
        self.idnet = IDnet(id_file_path).requires_grad_(False)#don't have to modify resolution
        # self.idnet.eval()
        # self.lndnet = LNDnet(lnd_file_path).requires_grad_(False)#modify resolution
        self.bgnet = BGnet(bg_file_path).requires_grad_(False)#don't have to modify resolution
        self.linear = nn.Identity()#nn.Linear(512, 384)#model s:384, model B: 768 #With listener 512
        # self.lstm = nn.LSTM(512, 512, 1 , batch_first = True)
        self.emo_proj=nn.Sequential(nn.Linear(25,64),
                                    # nn.Dropout(p=0.5)
                                    nn.LayerNorm(64, eps=1e-6)
                                    )
        self.id_proj=nn.Sequential(nn.Linear(512,1024),#64-128
                                    torch.nn.GELU(),
                                    torch.nn.Linear(1024, 64),
                                    nn.LayerNorm(64, eps=1e-6)
                                    # nn.Dropout(p=0.5)
                                    )
        self.emo_cond_linear=nn.Linear(512*2,512)
        self.encode_video=ConvBlock()
        self.encode_video_listener=ConvBlock()
        # self.sp_drop=DropPath(0.5)
        # self.lis_drop=DropPath(0.5)
        self.sp_drop=feature_drop_emb(0.2)
        self.lis_drop=feature_drop_emb(0.2)
        self.lis_id_drop=feature_drop_emb(0.1)
        if au_file_path_disfa is not None:
            # pdb.set_trace()
            self.au_net_DISFA=MEFL.MEFARG(num_classes=8,backbone='resnet18').requires_grad_(False)# resnet
            # self.au_net_DISFA.load_state_dict(torch.load(au_file_path_disfa, map_location='cuda')['state_dict'])
            # self.au_net_DISFA.eval()
            self.au_net_BP4D=MEFL.MEFARG(num_classes=12,backbone='resnet18').requires_grad_(False) #num_classes: 12   neighbor_num: 4
            # self.au_net_BP4D.load_state_dict(torch.load(au_file_path_bp4d, map_location='cuda')['state_dict'])
            # self.au_net_BP4D.eval()
        if emotion_file_path is not None:
            #input resolution is 256
            
            self.emo_net=EmoNet(n_expression=8).requires_grad_(False)#what is the resolution#.to(device)()
            
            state_dict = torch.load(emotion_file_path, map_location="cpu")
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            self.emo_net.load_state_dict(state_dict,strict=True)
            # self.emo_net.load_state_dict(torch.load(emotion_file_path, map_location='cuda'),strict=True)
            
            self.emo_net.eval()
        if detect_face is not None:
            device = "cuda:0"
            self.sfd_detector = SFDDetector(device)
        self.resolution = torch.nn.AdaptiveAvgPool2d((256, 256))
    def forward(self, id,listener_ref=None, listener_video_clip_last=None,speaker_Audio=None, lnd=None, emo_cond=None, return_id_feats=False,
                return_emo=False,expression=False):
        # pdb.set_trace()
        # if emo_cond is not None:
        #     listener_ref=emo_cond['listener_ref']
        # else:
        #     listener_ref=None
        # hidden_stats=emo_cond['hidden_stats']
        # past_emo=None#emo_cond['past_emo']
        # pdb.set_trace()
        # if len(id.shape)==4:
        #     if lnd is None:
        #         lnd = id

        #     id_feats = self.idnet(id)  # [B, 512]
        #     # lnd_feats = self.lndnet(lnd) # [B, 102]
        #     feats = self.linear(torch.cat([id_feats, lnd_feats], dim=1))   # [B, 512]
            
        #     if return_id_feats:
        #         return feats, id_feats
        #     return feats
        # else:  #video process
            # pdb.set_trace()
            
        if lnd is None:
            lnd = id
        # pdb.set_trace()
        b,v,c,w,h=id.shape
        b,v,c_l,w_l,h_l=lnd.shape
        # pdb.set_trace()
        
        video_input=id.permute(0,2,1,3,4)
        encode_fea=self.encode_video(video_input) #[b,128]
        # pdb.set_trace()
        
        id_feats,emo_fea,au_disfa,au_bp4d=self.forward_listener(id)
        id=id.reshape(b*v,c,w,h)
        lnd=lnd.reshape(b*v,c_l,w_l,h_l)
        
        # # pdb.set_trace()
        # self.idnet.eval()
        # id_feats = self.idnet(id)  # [B, 512]

        #emo_feature
        # pdb.set_trace()
        au=torch.cat([au_bp4d, au_disfa[:, 4:5], au_disfa[:,-2:]],dim=-1)  #[8*b,15]->[b,8,15]
        au=torch.reshape(au,[b,-1,15])

        #
        expression=emo_fea['expression']
        expression_soft=Func.softmax(expression,dim=1)
        valence=emo_fea['valence']
        arousal=emo_fea['arousal']
        emo=torch.cat([expression_soft,torch.unsqueeze(valence,dim=-1),torch.unsqueeze(arousal,dim=-1)],dim=-1)
        emo=torch.reshape(emo,[b,-1,10])
        current_emo=torch.cat([au,emo],dim=-1)


       
        lisenter_ref_id= self.idnet(listener_ref[:,0,:,:,:])
        lisenter_ref_id=self.lis_id_drop(lisenter_ref_id,training=self.training)
        # lisenter_ref_id=self.lis_drop(lisenter_ref_id,training=self.training)
        # lnd_feats = self.lndnet(lnd) # [B, 102]
        # pdb.set_trace()
        # lis_drop
        # print(listener_video_clip_last.shape)
        if listener_video_clip_last is not None:
            video_input_listner_last=listener_video_clip_last.permute(0,2,1,3,4)
            lisenter_encode_fea=self.encode_video_listener(video_input_listner_last)
            # print(encode_fea.shape)
            # print(lisenter_encode_fea.shape)
            feats = self.linear(torch.cat([self.emo_proj(current_emo).mean(-2),
                encode_fea,lisenter_encode_fea,self.id_proj(lisenter_ref_id)], dim=1))   # [B, 64+128+128+64=384] #
        else:
            feats = self.linear(torch.cat([self.id_proj(id_feats), self.emo_proj(current_emo).mean(-2),
                encode_fea,self.id_proj(lisenter_ref_id)], dim=1))   # [B, 64+128+128+64=384]

        # feats = self.linear(torch.cat([self.emo_proj(current_emo).mean(-2),
        #         encode_fea,self.id_proj(lisenter_ref_id)], dim=1))   # [B, 512]
        # feats = feats.reshape(b,v,-1)
        # id_feats=id_feats.reshape(b,v,-1)


        

        # if len(feats.shape)==3:
        #     feats=torch.nn.functional.avg_pool2d(feats, (feats.shape[1], 1), stride=(feats.shape[1], 1)).squeeze()
        #     if id_feats is not None:
        #         id_feats = torch.nn.functional.avg_pool2d(id_feats, (id_feats.shape[1], 1), stride=(id_feats.shape[1], 1))#.squeeze()
        # # if return_emo:
        # #     # pdb.set_trace()
        # #     # self.emo_net.eval()
        # #     emo_fea=self.emo_net(id)
        # #     # emo_fea
        # #     # self.au_net_DISFA.eval()
        # #     # self.au_net_BP4D.eval()
        # #     au_disfa=self.au_net_DISFA(id)[0]
        # #     au_bp4d=self.au_net_BP4D(id)[0]

        # #     return feats,emo_fea,au_disfa,au_bp4d,hidden_stats
        # listener_id_feats = self.idnet(listener_ref[:,0,:,:,:]) 
        hidden_stats=None
        # # if listener_ref != None and past_emo!=None:
        # #     #1 get listener id
        # #     listener_id_feats = self.idnet(listener_ref) 
        # #     # speaker_feat=torch.cat([id_feats,au_disfa,au_bp4d,])
        # #     # pdb.set_trace()
        # #     #2 past emotion with lstm
        # #     past_emo=torch.nn.functional.avg_pool2d(past_emo, (past_emo.shape[1], 1), stride=(past_emo.shape[1], 1))#.squeeze()
        # #     # pdb.set_trace()
        # #     listener_temporal_emo, hidden_stats = self.lstm(self.emo_proj(past_emo),hidden_stats)# (hidden_stats, hidden_stats))
        # #     # feats=self.emo_cond_linear(
        # #     #     torch.cat([listener_id_feats,listener_temporal_emo.squeeze(),feats],dim=-1))#fuse speaker+ listener condition
        # #     # listener_temporal_emo, hidden_stats = self.lstm(self.emo_proj(id_feats),hidden_stats)# (hidden_stats, hidden_stats))
        # #     feats=self.emo_cond_linear(
        # #         torch.cat([listener_temporal_emo.squeeze(),feats],dim=-1))#fuse speaker+ listener condition
        # #     id_feats=id_feats.squeeze()
        # # pdb.set_trace()
        return feats,hidden_stats,id_feats,lisenter_ref_id
        # if return_id_feats:
        #     return feats, id_feats
        # return feats
    def forward_listener(self, generated_listener,emo_cond=None,return_emo=False,detect_face=False):
        # pdb.set_trace()
        id=generated_listener
        if len(generated_listener.shape)==5:
            b,v,c,w,h=id.shape
            # b,v,c_l,w_l,h_l=lnd.shape

            id=id.reshape(b*v,c,w,h)
        elif len(generated_listener.shape)==4:
            b,c,w,h=id.shape
            
        # b,v,c,w,h=id.shape
        # # b,v,c_l,w_l,h_l=lnd.shape

        # id=id.reshape(b*v,c,w,h)
        # lnd=lnd.view(b*v,c_l,w_l,h_l)

        # pdb.set_trace()
        id_feats = self.idnet(id)  # [B, 512]
        # lnd_feats = self.lndnet(id) # [B, 102]
        # fea
        # feats = self.linear(torch.cat([id_feats, lnd_feats], dim=1))   # [B, 512]
        # feats = feats.view(b,v,-1)
        
        if len(generated_listener.shape)==5:
            id_feats=id_feats.reshape(b,v,-1)


            

            # if len(feats.shape)==3:
            #     feats=torch.nn.functional.avg_pool2d(feats, (feats.shape[1], 1), stride=(feats.shape[1], 1)).squeeze()
            if id_feats is not None:
                id_feats = torch.nn.functional.avg_pool2d(id_feats, (id_feats.shape[1], 1), stride=(id_feats.shape[1], 1)).squeeze()
        elif len(generated_listener.shape)==4:
            # pdb.set_trace()
            if id_feats is not None:
                id_feats = id_feats.mean(dim=0)
                # torch.nn.functional.avg_pool2d(id_feats, (id_feats.shape[0], 1), stride=(id_feats.shape[0], 1)).squeeze()
        if detect_face == True:
            # pdb.set_trace()
            detect_face=self.sfd_detector.detect_from_batch(128*(id/2+0.5))
            emo_id=torch.zeros_like(id)
            for index,frame in enumerate(id):
                    
                # detected_faces = self.sfd_detector.detect_from_image(frame)#[:, :, ::-1]) #BGR frame
                
                # for each f
                x_min, y_min, x_max, y_max,confidence = detect_face[index][0]
                
                cropped_face = frame[:,np.int(y_min):np.int(y_max), np.int(x_min):np.int(x_max)]
                # else:
                #     crop_frame = frame
                resized_frame = F.interpolate(cropped_face, size=(256, 256), mode='bilinear', align_corners=False)
                # resized_frame=self.resolution(cropped_face)
                emo_id[index]=resized_frame
                # crop_frame = cv2.resize(cropped_face,(256,256),interpolation = cv2.INTER_LANCZOS4)
                # id[index]=detected_faces

            # pdb.set_trace()
            with torch.no_grad():
                emo_fea=self.emo_net(emo_id)
            pdb.set_trace()
        else:
            emo_fea=self.emo_net(id)
        
        # expression_soft=F.softmax(emo_fea,dim=1)
        # emo_fea
        # self.au_net_DISFA.eval()
        # self.au_net_BP4D.eval()
        au_disfa=self.au_net_DISFA(id)[0]
        au_bp4d=self.au_net_BP4D(id)[0]



        # if listener_ref != None and past_emo!=None:
        #     #1 get listener id
        #     listener_id_feats = self.idnet(listener_ref) 
        #     speaker_feat=torch.cat([id_feats,au_disfa,au_bp4d,])

        #     #2 past emotion with lstm
        #     listener_temporal_emo, hidden_status = self.lstm(self.emo_proj(past_emo[:,-1]), hidden_status, 
        #                             hidden_status)
        #     feats=torch.cat([listener_id_feats,listener_temporal_emo,feats])#fuse speaker+ listener condition
        #     # au_bp4d=model_forward.au_bp4d
        #     # au_disfa=model_forward.au_disfa

        #     # # pdb.set_trace()
        #     # #
        #     # au=torch.cat([au_bp4d, au_disfa[:, 4:5], au_disfa[:,-2:]],dim=-1)  #[8*b,15]->[b,8,15]
        #     # au=torch.reshape(au,[len(generated_listener),-1,15])

        #     # #
        #     # emo=torch.cat([expression_soft,torch.unsqueeze(valence,dim=-1),torch.unsqueeze(arousal,dim=-1)],dim=-1)
        #     # emo=torch.reshape(emo,[len(generated_listener),-1,10])
        #     # #au loss
        #     # au_loss= mean_flat((au-emo_label[:,:,:15])**2)
        #     # emo_loss= mean_flat((emo-emo_label[:,:,15:])**2)
        return id_feats,emo_fea,au_disfa,au_bp4d
        # if return_id_feats:
        #     return feats, id_feats
        # return feats
    
    def id_forward(self, id):
        return self.idnet(id)  # [B, 512]
        
    def forward_with_id(self, id_feats, lnd):
        lnd_feats = self.lndnet(lnd) # [B, 102]
        feats = self.linear(torch.cat([id_feats, lnd_feats], dim=1))   # [B, 512]
        return feats
    # def forward_emo(self,input_img):

    def face_mask(self, bg, for_video=False):
        fg_mask = self.bgnet(bg, for_video=for_video) # [B, 1, 256, 256]
        return fg_mask



class CalculateAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V, mask):
        attention = torch.matmul(Q,torch.transpose(K, -1, -2))
        # use mask
        attention = attention.masked_fill_(mask, -1e9)
        attention = torch.softmax(attention / sqrt(Q.size(-1)), dim=-1)
        attention = torch.matmul(attention,V)
        return attention

class Multi_CrossAttention(nn.Module):
    """
    forward时，第一个参数用于计算query，第二个参数用于计算key和value
    """
    def __init__(self,hidden_size,all_head_size,head_num):
        super().__init__()
        self.hidden_size    = hidden_size       # 输入维度
        self.all_head_size  = all_head_size     # 输出维度
        self.num_heads      = head_num          # 注意头的数量
        self.h_size         = all_head_size // head_num

        assert all_head_size % head_num == 0

        # W_Q,W_K,W_V (hidden_size,all_head_size)
        self.linear_q = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_k = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_v = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_output = nn.Linear(all_head_size, hidden_size)

        # normalization
        self.norm = sqrt(all_head_size)

    def print(self):
        print(self.hidden_size,self.all_head_size)
        print(self.linear_k,self.linear_q,self.linear_v)
    
    def forward(self,x,y,z,attention_mask):
        """
        cross-attention: x,y是两个模型的隐藏层，将x作为q的输入，y作为k和v的输入
        """
        batch_size = x.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)

        # q_s: [batch_size, num_heads, seq_length, h_size]
        q_s = self.linear_q(x).view(batch_size, -1, self.num_heads, self.h_size).transpose(1,2)

        # k_s: [batch_size, num_heads, seq_length, h_size]
        k_s = self.linear_k(y).view(batch_size, -1, self.num_heads, self.h_size).transpose(1,2)

        # v_s: [batch_size, num_heads, seq_length, h_size]
        v_s = self.linear_v(z).view(batch_size, -1, self.num_heads, self.h_size).transpose(1,2)

        attention_mask = attention_mask.eq(0)

        attention = CalculateAttention()(q_s,k_s,v_s,attention_mask)
        # attention : [batch_size , seq_length , num_heads * h_size]
        attention = attention.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.h_size)
        
        # output : [batch_size , seq_length , hidden_size]
        output = self.linear_output(attention)

        return output
class DiffusionVideoAutoencoder_v3(nn.Module):
    def __init__(self, id_file_path, lnd_file_path, bg_file_path,au_file_path_disfa=None,
    au_file_path_bp4d=None,emotion_file_path=None, detect_face=True):
        super(DiffusionVideoAutoencoder_v3, self).__init__()
        self.idnet = IDnet(id_file_path).requires_grad_(False)#don't have to modify resolution
        # self.idnet.eval()
        # self.lndnet = LNDnet(lnd_file_path).requires_grad_(False)#modify resolution
        self.bgnet = BGnet(bg_file_path).requires_grad_(False)#don't have to modify resolution
        self.linear = nn.Identity()#nn.Linear(512, 384)#model s:384, model B: 768 #With listener 512
        # self.lstm = nn.LSTM(512, 512, 1 , batch_first = True)
        self.emo_proj=nn.Sequential(nn.Linear(25,128),
                                    # nn.Dropout(p=0.5)
                                    nn.LayerNorm(128, eps=1e-6)
                                    )
        self.id_proj=nn.Sequential(nn.Linear(512,1024),#64-128
                                    torch.nn.GELU(),
                                    torch.nn.Linear(1024, 64),
                                    nn.LayerNorm(64, eps=1e-6)
                                    # nn.Dropout(p=0.5)
                                    )
        self.audio_proj=nn.Linear(78, 128)


        '''second projection'''
        self.enc_proj_sec=nn.Sequential(nn.Linear(128,128),
                                    # nn.Dropout(p=0.5)
                                    nn.LayerNorm(128, eps=1e-6)
                                    )
        self.enc_proj_listener_sec=nn.Sequential(nn.Linear(128,128),
                                    # nn.Dropout(p=0.5)
                                    nn.LayerNorm(128, eps=1e-6)
                                    )
        self.audio_proj_sec=self.enc_proj=nn.Sequential(nn.Linear(128,128),
                                    # nn.Dropout(p=0.5)
                                    nn.LayerNorm(128, eps=1e-6)
                                    )
        '''end second projection'''
        self.shared_att=Multi_CrossAttention(128,128,head_num=4)#input is  sp_facial, sp_audio, li_facial_past
        # 定义多头注意力模块
        self.shared_att = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)

        

        
        self.emo_cond_linear=nn.Linear(512*2,512)
        self.encode_video=ConvBlock()  #output is 128
        self.encode_video_listener=ConvBlock()
        # self.sp_drop=DropPath(0.5)
        # self.lis_drop=DropPath(0.5)
        self.sp_drop=feature_drop_emb(0.2)
        self.lis_drop=feature_drop_emb(0.2)
        self.lis_id_drop=feature_drop_emb(0.1)
        if au_file_path_disfa is not None:
            # pdb.set_trace()
            self.au_net_DISFA=MEFL.MEFARG(num_classes=8,backbone='resnet18').requires_grad_(False)# resnet
            # self.au_net_DISFA.load_state_dict(torch.load(au_file_path_disfa, map_location='cuda')['state_dict'])
            # self.au_net_DISFA.eval()
            self.au_net_BP4D=MEFL.MEFARG(num_classes=12,backbone='resnet18').requires_grad_(False) #num_classes: 12   neighbor_num: 4
            # self.au_net_BP4D.load_state_dict(torch.load(au_file_path_bp4d, map_location='cuda')['state_dict'])
            # self.au_net_BP4D.eval()
        if emotion_file_path is not None:
            #input resolution is 256
            
            self.emo_net=EmoNet(n_expression=8).requires_grad_(False)#what is the resolution#.to(device)()
            
            state_dict = torch.load(emotion_file_path, map_location="cpu")
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            self.emo_net.load_state_dict(state_dict,strict=True)
            # self.emo_net.load_state_dict(torch.load(emotion_file_path, map_location='cuda'),strict=True)
            
            self.emo_net.eval()
        if detect_face is not None:
            device = "cuda:0"
            self.sfd_detector = SFDDetector(device)
        self.resolution = torch.nn.AdaptiveAvgPool2d((256, 256))
    def forward(self, id,listener_ref=None, listener_video_clip_last=None, speaker_Audio=None, lnd=None, emo_cond=None, return_id_feats=False,
                return_emo=False,expression=False):
        # pdb.set_trace()
        # if emo_cond is not None:
        #     listener_ref=emo_cond['listener_ref']
        # else:
        #     listener_ref=None
        # hidden_stats=emo_cond['hidden_stats']
        # past_emo=None#emo_cond['past_emo']
        # pdb.set_trace()
        # if len(id.shape)==4:
        #     if lnd is None:
        #         lnd = id

        #     id_feats = self.idnet(id)  # [B, 512]
        #     # lnd_feats = self.lndnet(lnd) # [B, 102]
        #     feats = self.linear(torch.cat([id_feats, lnd_feats], dim=1))   # [B, 512]
            
        #     if return_id_feats:
        #         return feats, id_feats
        #     return feats
        # else:  #video process
            # pdb.set_trace()
            
        if lnd is None:
            lnd = id
        # pdb.set_trace()
        b,v,c,w,h=id.shape
        b,v,c_l,w_l,h_l=lnd.shape
        # pdb.set_trace()
        
        video_input=id.permute(0,2,1,3,4)
        encode_fea=self.encode_video(video_input) #[b,128]
        # pdb.set_trace()
        
        id_feats,emo_fea,au_disfa,au_bp4d=self.forward_listener(id)
        id=id.reshape(b*v,c,w,h)
        lnd=lnd.reshape(b*v,c_l,w_l,h_l)
        
        # # pdb.set_trace()
        # self.idnet.eval()
        # id_feats = self.idnet(id)  # [B, 512]

        #emo_feature
        # pdb.set_trace()
        au=torch.cat([au_bp4d, au_disfa[:, 4:5], au_disfa[:,-2:]],dim=-1)  #[8*b,15]->[b,8,15]
        au=torch.reshape(au,[b,-1,15])

        #
        expression=emo_fea['expression']
        expression_soft=Func.softmax(expression,dim=1)
        valence=emo_fea['valence']
        arousal=emo_fea['arousal']
        emo=torch.cat([expression_soft,torch.unsqueeze(valence,dim=-1),torch.unsqueeze(arousal,dim=-1)],dim=-1)
        emo=torch.reshape(emo,[b,-1,10])
        current_emo=torch.cat([au,emo],dim=-1)


       
        
        lisenter_ref_id= self.idnet(listener_ref[:,0,:,:,:])
        lisenter_ref_id=self.lis_id_drop(lisenter_ref_id,training=self.training)
        # lisenter_ref_id=self.lis_drop(lisenter_ref_id,training=self.training)
        # lnd_feats = self.lndnet(lnd) # [B, 102]
        # pdb.set_trace()
        # lis_drop
        # print(listener_video_clip_last.shape)
        
        if listener_video_clip_last is not None:
            '''obtain listener feature'''
            video_input_listner_last=listener_video_clip_last.permute(0,2,1,3,4)
            lisenter_encode_fea=self.encode_video_listener(video_input_listner_last)


            '''multi-modality fusion'''
            
            audio_feat=self.audio_proj(speaker_Audio).mean(dim=1)
            print(audio_feat.shape)
            shared_feat.squeeze(),attention_weight=self.shared_att(query=encode_fea.unsqueeze(1),key=lisenter_encode_fea.unsqueeze(1),value=audio_feat.unsqueeze(1))

            # attn_output, attn_weights = cross_attention(query=query, key=key, value=value)
            encode_fea=self.enc_proj_sec(encode_fea)
            lisenter_encode_fea=self.enc_proj_listener_sec(lisenter_encode_fea)
            audio_feat=self.audio_proj_sec(audio_feat)

            '''fusion end'''
            # print(encode_fea.shape)
            # print(lisenter_encode_fea.shape)
            print('shared_feat',shared_feat.shape)
            print('encode_fea',encode_fea.shape)
            print('lisenter_encode_fea',lisenter_encode_fea.shape)
            print('lisenter_ref_id',lisenter_ref_id.shape)
            feats = self.linear(torch.cat([shared_feat,
                encode_fea,lisenter_encode_fea,self.id_proj(lisenter_ref_id)], dim=1))   # [B, 64+128+128+64=384] #
        else:
            feats = self.linear(torch.cat([self.id_proj(id_feats), self.emo_proj(current_emo).mean(-2),
                encode_fea,self.id_proj(lisenter_ref_id)], dim=1))   # [B, 64+128+128+64=384]

        # feats = self.linear(torch.cat([self.emo_proj(current_emo).mean(-2),
        #         encode_fea,self.id_proj(lisenter_ref_id)], dim=1))   # [B, 512]
        # feats = feats.reshape(b,v,-1)
        # id_feats=id_feats.reshape(b,v,-1)


        

        # if len(feats.shape)==3:
        #     feats=torch.nn.functional.avg_pool2d(feats, (feats.shape[1], 1), stride=(feats.shape[1], 1)).squeeze()
        #     if id_feats is not None:
        #         id_feats = torch.nn.functional.avg_pool2d(id_feats, (id_feats.shape[1], 1), stride=(id_feats.shape[1], 1))#.squeeze()
        # # if return_emo:
        # #     # pdb.set_trace()
        # #     # self.emo_net.eval()
        # #     emo_fea=self.emo_net(id)
        # #     # emo_fea
        # #     # self.au_net_DISFA.eval()
        # #     # self.au_net_BP4D.eval()
        # #     au_disfa=self.au_net_DISFA(id)[0]
        # #     au_bp4d=self.au_net_BP4D(id)[0]

        # #     return feats,emo_fea,au_disfa,au_bp4d,hidden_stats
        # listener_id_feats = self.idnet(listener_ref[:,0,:,:,:]) 
        hidden_stats=None
        # # if listener_ref != None and past_emo!=None:
        # #     #1 get listener id
        # #     listener_id_feats = self.idnet(listener_ref) 
        # #     # speaker_feat=torch.cat([id_feats,au_disfa,au_bp4d,])
        # #     # pdb.set_trace()
        # #     #2 past emotion with lstm
        # #     past_emo=torch.nn.functional.avg_pool2d(past_emo, (past_emo.shape[1], 1), stride=(past_emo.shape[1], 1))#.squeeze()
        # #     # pdb.set_trace()
        # #     listener_temporal_emo, hidden_stats = self.lstm(self.emo_proj(past_emo),hidden_stats)# (hidden_stats, hidden_stats))
        # #     # feats=self.emo_cond_linear(
        # #     #     torch.cat([listener_id_feats,listener_temporal_emo.squeeze(),feats],dim=-1))#fuse speaker+ listener condition
        # #     # listener_temporal_emo, hidden_stats = self.lstm(self.emo_proj(id_feats),hidden_stats)# (hidden_stats, hidden_stats))
        # #     feats=self.emo_cond_linear(
        # #         torch.cat([listener_temporal_emo.squeeze(),feats],dim=-1))#fuse speaker+ listener condition
        # #     id_feats=id_feats.squeeze()
        # # pdb.set_trace()
        return feats,hidden_stats,id_feats,lisenter_ref_id
        # if return_id_feats:
        #     return feats, id_feats
        # return feats
    def forward_listener(self, generated_listener,emo_cond=None,return_emo=False,detect_face=False):
        # pdb.set_trace()
        id=generated_listener
        if len(generated_listener.shape)==5:
            b,v,c,w,h=id.shape
            # b,v,c_l,w_l,h_l=lnd.shape

            id=id.reshape(b*v,c,w,h)
        elif len(generated_listener.shape)==4:
            b,c,w,h=id.shape
            
        # b,v,c,w,h=id.shape
        # # b,v,c_l,w_l,h_l=lnd.shape

        # id=id.reshape(b*v,c,w,h)
        # lnd=lnd.view(b*v,c_l,w_l,h_l)

        # pdb.set_trace()
        id_feats = self.idnet(id)  # [B, 512]
        # lnd_feats = self.lndnet(id) # [B, 102]
        # fea
        # feats = self.linear(torch.cat([id_feats, lnd_feats], dim=1))   # [B, 512]
        # feats = feats.view(b,v,-1)
        
        if len(generated_listener.shape)==5:
            id_feats=id_feats.reshape(b,v,-1)


            

            # if len(feats.shape)==3:
            #     feats=torch.nn.functional.avg_pool2d(feats, (feats.shape[1], 1), stride=(feats.shape[1], 1)).squeeze()
            if id_feats is not None:
                id_feats = torch.nn.functional.avg_pool2d(id_feats, (id_feats.shape[1], 1), stride=(id_feats.shape[1], 1)).squeeze()
        elif len(generated_listener.shape)==4:
            # pdb.set_trace()
            if id_feats is not None:
                id_feats = id_feats.mean(dim=0)
                # torch.nn.functional.avg_pool2d(id_feats, (id_feats.shape[0], 1), stride=(id_feats.shape[0], 1)).squeeze()
        if detect_face == True:
            # pdb.set_trace()
            detect_face=self.sfd_detector.detect_from_batch(128*(id/2+0.5))
            emo_id=torch.zeros_like(id)
            for index,frame in enumerate(id):
                    
                # detected_faces = self.sfd_detector.detect_from_image(frame)#[:, :, ::-1]) #BGR frame
                
                # for each f
                x_min, y_min, x_max, y_max,confidence = detect_face[index][0]
                
                cropped_face = frame[:,np.int(y_min):np.int(y_max), np.int(x_min):np.int(x_max)]
                # else:
                #     crop_frame = frame
                resized_frame = F.interpolate(cropped_face, size=(256, 256), mode='bilinear', align_corners=False)
                # resized_frame=self.resolution(cropped_face)
                emo_id[index]=resized_frame
                # crop_frame = cv2.resize(cropped_face,(256,256),interpolation = cv2.INTER_LANCZOS4)
                # id[index]=detected_faces

            # pdb.set_trace()
            with torch.no_grad():
                emo_fea=self.emo_net(emo_id)
            pdb.set_trace()
        else:
            emo_fea=self.emo_net(id)
        
        # expression_soft=F.softmax(emo_fea,dim=1)
        # emo_fea
        # self.au_net_DISFA.eval()
        # self.au_net_BP4D.eval()
        au_disfa=self.au_net_DISFA(id)[0]
        au_bp4d=self.au_net_BP4D(id)[0]



        # if listener_ref != None and past_emo!=None:
        #     #1 get listener id
        #     listener_id_feats = self.idnet(listener_ref) 
        #     speaker_feat=torch.cat([id_feats,au_disfa,au_bp4d,])

        #     #2 past emotion with lstm
        #     listener_temporal_emo, hidden_status = self.lstm(self.emo_proj(past_emo[:,-1]), hidden_status, 
        #                             hidden_status)
        #     feats=torch.cat([listener_id_feats,listener_temporal_emo,feats])#fuse speaker+ listener condition
        #     # au_bp4d=model_forward.au_bp4d
        #     # au_disfa=model_forward.au_disfa

        #     # # pdb.set_trace()
        #     # #
        #     # au=torch.cat([au_bp4d, au_disfa[:, 4:5], au_disfa[:,-2:]],dim=-1)  #[8*b,15]->[b,8,15]
        #     # au=torch.reshape(au,[len(generated_listener),-1,15])

        #     # #
        #     # emo=torch.cat([expression_soft,torch.unsqueeze(valence,dim=-1),torch.unsqueeze(arousal,dim=-1)],dim=-1)
        #     # emo=torch.reshape(emo,[len(generated_listener),-1,10])
        #     # #au loss
        #     # au_loss= mean_flat((au-emo_label[:,:,:15])**2)
        #     # emo_loss= mean_flat((emo-emo_label[:,:,15:])**2)
        return id_feats,emo_fea,au_disfa,au_bp4d
        # if return_id_feats:
        #     return feats, id_feats
        # return feats
    
    def id_forward(self, id):
        return self.idnet(id)  # [B, 512]
        
    def forward_with_id(self, id_feats, lnd):
        lnd_feats = self.lndnet(lnd) # [B, 102]
        feats = self.linear(torch.cat([id_feats, lnd_feats], dim=1))   # [B, 512]
        return feats
    # def forward_emo(self,input_img):

    def face_mask(self, bg, for_video=False):
        fg_mask = self.bgnet(bg, for_video=for_video) # [B, 1, 256, 256]
        return fg_mask

# class facial_conclass(nn.Module):
#     """
#     Diffusion model with a Transformer backbone.
#     """
#     def __init__():
#         a=1
#     def forward(original_x):
#         a=1

#         return facial_condition
def get_1d_sincos_temp_embed(embed_dim, length):
    pos = torch.arange(0, length).unsqueeze(1)
    return get_1d_sincos_pos_embed_from_grid(embed_dim, pos)

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0]) 
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1]) 

    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega 

    pos = pos.reshape(-1)  
    out = np.einsum('m,d->md', pos, omega) 

    emb_sin = np.sin(out) 
    emb_cos = np.cos(out) 

    emb = np.concatenate([emb_sin, emb_cos], axis=1) 
    return emb


#################################################################################
#                                   Latte Configs                                  #
#################################################################################

def Latte_XL_2(**kwargs):
    return Latte(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def Latte_XL_4(**kwargs):
    return Latte(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def Latte_XL_8(**kwargs):
    return Latte(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def Latte_L_2(**kwargs):
    return Latte(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def Latte_L_4(**kwargs):
    return Latte(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def Latte_L_8(**kwargs):
    return Latte(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def Latte_B_2(**kwargs):
    return Latte(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def Latte_B_4(**kwargs):
    return Latte(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def Latte_B_8(**kwargs):
    return Latte(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def Latte_S_2(**kwargs):
    return Latte(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def Latte_S_4(**kwargs):
    return Latte(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def Latte_S_8(**kwargs):
    return Latte(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


Latte_models = {
    'Latte-XL/2': Latte_XL_2,  'Latte-XL/4': Latte_XL_4,  'Latte-XL/8': Latte_XL_8,
    'Latte-L/2':  Latte_L_2,   'Latte-L/4':  Latte_L_4,   'Latte-L/8':  Latte_L_8,
    'Latte-B/2':  Latte_B_2,   'Latte-B/4':  Latte_B_4,   'Latte-B/8':  Latte_B_8,
    'Latte-S/2':  Latte_S_2,   'Latte-S/4':  Latte_S_4,   'Latte-S/8':  Latte_S_8,
}

if __name__ == '__main__':

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    img = torch.randn(3, 16, 4, 32, 32).to(device)
    t = torch.tensor([1, 2, 3]).to(device)
    y = torch.tensor([1, 2, 3]).to(device)
    network = Latte_XL_2().to(device)
    from thop import profile 
    flops, params = profile(network, inputs=(img, t))
    print('FLOPs = ' + str(flops/1000**3) + 'G')
    print('Params = ' + str(params/1000**2) + 'M')
    # y_embeder = LabelEmbedder(num_classes=101, hidden_size=768, dropout_prob=0.5).to(device)
    # lora.mark_only_lora_as_trainable(network)
    # out = y_embeder(y, True)
    # out = network(img, t, y)
    # print(out.shape)