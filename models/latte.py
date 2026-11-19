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

from models.emo_fr_net.model_irse import IDnet
from models.emo_fr_net.seg_model_2 import BGnet
from models.emo_fr_net.emo_net import EmoNet

from models.emo_fr_net.au_model import MEFL

from face_alignment.detection.sfd.sfd_detector import SFDDetector
from math import sqrt

try:
    import xformers
    import xformers.ops
except:
    XFORMERS_IS_AVAILBLE = False

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

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

class conditional_zero_masking(nn.Module):
    def __init__(self,  v_dropout_prob=0.05,id_drop_prob=0.05,both_drop_prob=0.05):
        super().__init__()
        self.v_dropout_prob=v_dropout_prob
        self.id_drop_prob=id_drop_prob
        self.both_drop_prob=both_drop_prob

    def forward(self, image: torch.Tensor,listener_image: torch.Tensor, embedding: torch.Tensor,training=False):

        if training:
            T1=self.v_dropout_prob
            T2=self.v_dropout_prob + self.id_drop_prob
            T3=self.v_dropout_prob + self.id_drop_prob+self.both_drop_prob
            batch_size = image.size(0)

            random_probs = torch.rand(batch_size, device=image.device)

            masked_image = image.clone()
            masked_listener_past=listener_image.clone()
            masked_embedding = embedding.clone()

            mask_image_only = (random_probs >= 0) & (random_probs < T1)
            if mask_image_only.any():
                masked_image[mask_image_only] = 0
                masked_listener_past[mask_image_only]=0
            mask_embedding_only = (random_probs >= T1) & (random_probs < T2)
            if mask_embedding_only.any():
                masked_embedding[mask_embedding_only] = 0

            mask_both = (random_probs >= T2) & (random_probs < T3)
            if mask_both.any():
                masked_image[mask_both] = 0
                masked_listener_past[mask_both]=0
                masked_embedding[mask_both] = 0

        else:
            masked_image=image
            masked_listener_past=listener_image
            masked_embedding=embedding
        return masked_image, masked_listener_past,masked_embedding
class image_drop_emb(nn.Module):
    def __init__(self,  dropout_prob=0.0):
        super().__init__()
        self.dropout_prob=dropout_prob

    def forward(self,image,training=False):
        use_dropout = self.dropout_prob > 0
        if (training and use_dropout):
            drop_ids = torch.rand((image.shape[0],1,1,1,1), device=image.device) < self.dropout_prob
            image = torch.where(drop_ids, torch.zeros_like(image), image)
        return image

class feature_drop_emb(nn.Module):
    def __init__(self,  dropout_prob=0.0):
        super().__init__()
        self.dropout_prob=dropout_prob

    def forward(self,image,training=False):
        use_dropout = self.dropout_prob > 0
        if (training and use_dropout):
            drop_ids = torch.rand((image.shape[0],1), device=image.device) < self.dropout_prob
            image = torch.where(drop_ids, torch.zeros_like(image), image)
        return image

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

            self.facial_condition = DiffusionVideoAutoencoder_v5("./pretrained_models/model_ir_se50.pth", "./pretrained_models/mobilenet_224_model_best_gdconv_external.pth.tar", "./pretrained_models/79999_iter.pth"
                    ,emotion_file_path='./pretrained_models/emonet_8.pth', au_file_path_disfa='./pretrained_models/MEFARG_swin_base_DISFA_fold1.pth',au_file_path_bp4d='./pretrained_models/MEFARG_swin_base_BP4D_fold1.pth')

            self.sp_drop_operation=image_drop_emb(0.2)
            self.lis_drop_operation=image_drop_emb(0.2)
            self.feat_drop_operation=feature_drop_emb(0.1)
        if self.extras == 78: # timestep + text_embedding
            self.text_embedding_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(77 * 768, hidden_size, bias=True)
        )

        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        self.temp_embed = nn.Parameter(torch.zeros(1, num_frames, hidden_size), requires_grad=False)
        self.hidden_size =  hidden_size

        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, attention_mode=attention_mode) for _ in range(depth)
        ])

        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        temp_embed = get_1d_sincos_temp_embed(self.temp_embed.shape[-1], self.temp_embed.shape[-2])
        self.temp_embed.data.copy_(torch.from_numpy(temp_embed).float().unsqueeze(0))

        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        if self.extras == 2:
            nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

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
        x = self.x_embedder(x) + self.pos_embed
        t = self.t_embedder(t, use_fp16=use_fp16)
        timestep_spatial = repeat(t, 'n d -> (n c) d', c=self.temp_embed.shape[1])
        timestep_temp = repeat(t, 'n d -> (n c) d', c=self.pos_embed.shape[1])

        if self.extras == 2:
            y = self.y_embedder(y, self.training)
            y_spatial = repeat(y, 'n d -> (n c) d', c=self.temp_embed.shape[1])
            y_temp = repeat(y, 'n d -> (n c) d', c=self.pos_embed.shape[1])
        elif self.extras == 3:
            facial_cond,hidden_stats,id_feats,listener_id_feats=self.facial_condition(original_x,listener_ref=listener_ref,
                    listener_video_clip_last=listener_video_clip_last,speaker_Audio=sp_audio_clip)
            facial_spatial = repeat(facial_cond, 'n d -> (n c) d', c=self.temp_embed.shape[1])
            facial_temp = repeat(facial_cond, 'n d -> (n c) d', c=self.pos_embed.shape[1])
        elif self.extras == 78:
            text_embedding = self.text_embedding_projection(text_embedding.reshape(batches, -1))
            text_embedding_spatial = repeat(text_embedding, 'n d -> (n c) d', c=self.temp_embed.shape[1])
            text_embedding_temp = repeat(text_embedding, 'n d -> (n c) d', c=self.pos_embed.shape[1])
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
            x  = spatial_block(x, c)

            x = rearrange(x, '(b f) t d -> (b t) f d', b=batches)
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
        if self.extras == 2:
            c = timestep_spatial + y_spatial
        elif self.extras == 3:
            c = timestep_spatial + facial_spatial
        else:
            c = timestep_spatial
        x = self.final_layer(x, c)
        x = self.unpatchify(x)
        x = rearrange(x, '(b f) c h w -> b f c h w', b=batches)
        return x

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

        y = x.mean(dim=-1).mean(dim=-1).mean(dim=-1)
        return y

class CalculateAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V, mask):
        attention = torch.matmul(Q,torch.transpose(K, -1, -2))
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

        self.linear_q = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_k = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_v = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_output = nn.Linear(all_head_size, hidden_size)

        self.norm = sqrt(all_head_size)

    def print(self):
        print(self.hidden_size,self.all_head_size)
        print(self.linear_k,self.linear_q,self.linear_v)

    def forward(self,x,y,z,attention_mask):
        """
        cross-attention: x,y是两个模型的隐藏层，将x作为q的输入，y作为k和v的输入
        """
        batch_size = x.size(0)

        q_s = self.linear_q(x).view(batch_size, -1, self.num_heads, self.h_size).transpose(1,2)

        k_s = self.linear_k(y).view(batch_size, -1, self.num_heads, self.h_size).transpose(1,2)

        v_s = self.linear_v(z).view(batch_size, -1, self.num_heads, self.h_size).transpose(1,2)

        attention_mask = attention_mask.eq(0)

        attention = CalculateAttention()(q_s,k_s,v_s,attention_mask)
        attention = attention.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.h_size)

        output = self.linear_output(attention)

        return output

class DiffusionVideoAutoencoder_v5(nn.Module):
    def __init__(self, id_file_path, lnd_file_path, bg_file_path,au_file_path_disfa=None,
    au_file_path_bp4d=None,emotion_file_path=None, detect_face=True,i_drop_rate=0.05,t_drop_rate=0.05,ti_drop_rate=0.05):
        super(DiffusionVideoAutoencoder_v5, self).__init__()
        self.idnet = IDnet(id_file_path).requires_grad_(False)#don't have to modify resolution
        self.bgnet = BGnet(bg_file_path).requires_grad_(False)#don't have to modify resolution
        self.linear = nn.Identity()#nn.Linear(512, 384)#model s:384, model B: 768 #With listener 512
        self.emo_proj=nn.Sequential(nn.Linear(25,128),
                                    nn.LayerNorm(128, eps=1e-6)
                                    )
        self.id_proj=nn.Sequential(nn.Linear(512,1024),#64-128
                                    torch.nn.GELU(),
                                    torch.nn.Linear(1024, 64),
                                    nn.LayerNorm(64, eps=1e-6)
                                    )
        self.audio_proj=nn.Linear(78, 128)

        '''second projection'''

        self.id_proj=nn.Sequential(nn.Linear(512,1024),#64-128
                                    torch.nn.GELU(),
                                    torch.nn.Linear(1024, 128),
                                    nn.LayerNorm(128, eps=1e-6)
                                    )
        self.enc_proj_sec=nn.Sequential(nn.Linear(128,256),
                                    nn.LayerNorm(256, eps=1e-6)
                                    )
        self.enc_proj_listener_sec=nn.Sequential(nn.Linear(128,256),
                                    nn.LayerNorm(256, eps=1e-6)
                                    )
        self.audio_proj_sec=self.enc_proj=nn.Sequential(nn.Linear(128,128),
                                    nn.LayerNorm(128, eps=1e-6)
                                    )
        '''end second projection'''
        self.shared_att=Multi_CrossAttention(128,128,head_num=4)#input is  sp_facial, sp_audio, li_facial_past
        self.shared_att = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
        self.att_linear=nn.Linear(128,64)

        self.gating=nn.LeakyReLU()

        self.emo_cond_linear=nn.Linear(512*2,512)
        self.encode_video=ConvBlock()  #output is 128
        self.encode_video_listener=ConvBlock()
        self.sp_drop=feature_drop_emb(0.2)
        self.lis_drop=feature_drop_emb(0.2)
        self.lis_id_drop=feature_drop_emb(0.1)
        self.both_drop=conditional_zero_masking(0.05,0.05,0.05)
        if au_file_path_disfa is not None:
            self.au_net_DISFA=MEFL.MEFARG(num_classes=8,backbone='resnet18').requires_grad_(False)# resnet
            self.au_net_BP4D=MEFL.MEFARG(num_classes=12,backbone='resnet18').requires_grad_(False) #num_classes: 12   neighbor_num: 4
        if emotion_file_path is not None:

            self.emo_net=EmoNet(n_expression=8).requires_grad_(False)#what is the resolution#.to(device)()

            state_dict = torch.load(emotion_file_path, map_location="cpu")
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            self.emo_net.load_state_dict(state_dict,strict=True)

            self.emo_net.eval()
        if detect_face is not None:
            device = f"cuda:{torch.cuda.current_device()}"
            self.sfd_detector = SFDDetector(device)
        self.resolution = torch.nn.AdaptiveAvgPool2d((256, 256))
    def forward(self, id,listener_ref=None, listener_video_clip_last=None, speaker_Audio=None, lnd=None, emo_cond=None, return_id_feats=False,
                return_emo=False,expression=False):

        if lnd is None:
            lnd = id
        b,v,c,w,h=id.shape
        b,v,c_l,w_l,h_l=lnd.shape

        '''obtain listener identity'''
        lisenter_ref_id= self.idnet(listener_ref[:,0,:,:,:])
        '''drop video and id'''
        id,listener_video_clip_last,lisenter_ref_id=self.both_drop(id,listener_video_clip_last,lisenter_ref_id,training=self.training)

        video_input=id.permute(0,2,1,3,4)
        encode_fea=self.encode_video(video_input) #[b,128]

        id_feats,emo_fea,au_disfa,au_bp4d=self.forward_listener(id)
        id=id.reshape(b*v,c,w,h)
        lnd=lnd.reshape(b*v,c_l,w_l,h_l)

        au=torch.cat([au_bp4d, au_disfa[:, 4:5], au_disfa[:,-2:]],dim=-1)  #[8*b,15]->[b,8,15]
        au=torch.reshape(au,[b,-1,15])

        expression=emo_fea['expression']
        expression_soft=Func.softmax(expression,dim=1)
        valence=emo_fea['valence']
        arousal=emo_fea['arousal']
        emo=torch.cat([expression_soft,torch.unsqueeze(valence,dim=-1),torch.unsqueeze(arousal,dim=-1)],dim=-1)
        emo=torch.reshape(emo,[b,-1,10])
        current_emo=torch.cat([au,emo],dim=-1)

        if listener_video_clip_last is not None:
            '''obtain listener feature'''
            video_input_listner_last=listener_video_clip_last.permute(0,2,1,3,4)
            lisenter_encode_fea=self.encode_video_listener(video_input_listner_last)

            '''multi-modality fusion'''

            audio_feat=self.audio_proj(speaker_Audio).mean(dim=1)
            shared_feat,attention_weight=self.shared_att(query=encode_fea.unsqueeze(1),key=audio_feat.unsqueeze(1),value=audio_feat.unsqueeze(1))
            shared_feat=shared_feat.squeeze()

            encode_fea=encode_fea-shared_feat
            audio_feat=audio_feat-shared_feat

            encode_fea=self.gating(encode_fea)
            audio_feat=self.gating(audio_feat)

            encode_fea=self.enc_proj_sec(encode_fea)
            lisenter_encode_fea=self.enc_proj_listener_sec(lisenter_encode_fea)
            audio_feat=self.audio_proj_sec(audio_feat)

            '''fusion end'''
            feats = self.linear(torch.cat([audio_feat,
                encode_fea,lisenter_encode_fea,self.id_proj(lisenter_ref_id)], dim=1))   # [B, 64+128+128+64=384] #
        else:
            feats = self.linear(torch.cat([self.id_proj(id_feats), self.emo_proj(current_emo).mean(-2),
                encode_fea,self.id_proj(lisenter_ref_id)], dim=1))   # [B, 64+128+128+64=384]

        hidden_stats=None
        return feats,hidden_stats,id_feats,lisenter_ref_id
    def forward_listener(self, generated_listener,emo_cond=None,return_emo=False,detect_face=False):
        id=generated_listener
        if len(generated_listener.shape)==5:
            b,v,c,w,h=id.shape

            id=id.reshape(b*v,c,w,h)
        elif len(generated_listener.shape)==4:
            b,c,w,h=id.shape

        id_feats = self.idnet(id)  # [B, 512]

        if len(generated_listener.shape)==5:
            id_feats=id_feats.reshape(b,v,-1)

            if id_feats is not None:
                id_feats = torch.nn.functional.avg_pool2d(id_feats, (id_feats.shape[1], 1), stride=(id_feats.shape[1], 1)).squeeze()
        elif len(generated_listener.shape)==4:
            if id_feats is not None:
                id_feats = id_feats.mean(dim=0)
        if detect_face == True:
            detect_face=self.sfd_detector.detect_from_batch(128*(id/2+0.5))
            emo_id=torch.zeros_like(id)
            for index,frame in enumerate(id):

                x_min, y_min, x_max, y_max,confidence = detect_face[index][0]

                cropped_face = frame[:,int(y_min):int(y_max), int(x_min):int(x_max)]
                resized_frame = F.interpolate(cropped_face, size=(256, 256), mode='bilinear', align_corners=False)
                emo_id[index]=resized_frame

            with torch.no_grad():
                emo_fea=self.emo_net(emo_id)
        else:
            emo_fea=self.emo_net(id)

        au_disfa=self.au_net_DISFA(id)[0]
        au_bp4d=self.au_net_BP4D(id)[0]

        return id_feats,emo_fea,au_disfa,au_bp4d

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
