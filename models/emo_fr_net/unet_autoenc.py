from enum import Enum

import torch
from torch import Tensor
from torch import nn
from torch.nn.functional import silu
from torchvision import transforms

from .latentnet import *
from .unet import *
# from choices import *
from model.model_irse import IDnet
from model.basenet import LNDnet
from model.seg_model_2 import BGnet
from model.emo_net import EmoNet
import pdb
from model.au_model import MEFL
@dataclass
class BeatGANsAutoencConfig(BeatGANsUNetConfig):
    # number of style channels
    enc_out_channels: int = 512
    enc_attn_resolutions: Tuple[int] = None
    enc_pool: str = 'depthconv'
    enc_num_res_block: int = 2
    enc_channel_mult: Tuple[int] = None
    enc_grad_checkpoint: bool = False
    latent_net_conf: MLPSkipNetConfig = None

    def make_model(self):
        return BeatGANsAutoencModel(self)


class BeatGANsAutoencModel(BeatGANsUNetModel):
    def __init__(self, conf: BeatGANsAutoencConfig):
        super().__init__(conf)
        self.conf = conf

        # having only time, cond
        self.time_embed = TimeStyleSeperateEmbed(
            time_channels=conf.model_channels,
            time_out_channels=conf.embed_channels,
        )

        self.encoder = DiffusionVideoAutoencoder("./pretrained_models/model_ir_se50.pth", "./pretrained_models/mobilenet_224_model_best_gdconv_external.pth.tar", "./pretrained_models/79999_iter.pth"
                    ,emotion_file_path='./pretrained_models/emonet_8.pth', au_file_path_disfa='./pretrained_models/MEFARG_swin_base_DISFA_fold1.pth',au_file_path_bp4d='./pretrained_models/MEFARG_swin_base_BP4D_fold1.pth')
        if conf.latent_net_conf is not None:
            self.latent_net = conf.latent_net_conf.make_model()

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        assert self.conf.is_stochastic
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def sample_z(self, n: int, device):
        assert self.conf.is_stochastic
        return torch.randn(n, self.conf.enc_out_channels, device=device)

    def noise_to_cond(self, noise: Tensor):
        raise NotImplementedError()
        assert self.conf.noise_net_conf is not None
        return self.noise_net.forward(noise)

    def encode(self, x,emo_cond=None, return_id_feats=False,return_emo=False):
        # pdb.set_trace()
        id_listener=None
        emo_fea,au_disfa,au_bp4d=None,None,None
        if return_id_feats:
            cond, id_feats = self.encoder.forward(x,emo_cond=emo_cond, return_id_feats=return_id_feats)
            # if listener_ref!= None:
            #     cond_l, id_feats_l = self.encoder.forward(listener_ref, return_id_feats=return_id_feats)
        # elif return_emo:
        #     cond = self.encoder.forward(x,return_emo=True)
        else:
            # cond = self.encoder.forward(x)
            cond,hidden_stats,id_feats,id_feats_l = self.encoder.forward(x,emo_cond=emo_cond,return_emo=True)
            id_feats = None
        cond_other = None
        
        return {'cond': cond, 'id_listener':id_feats_l, 'id_feats': id_feats,'hidden_stats':hidden_stats,
                    
                    'cond_other': cond_other,'emo_fea': emo_fea,'au_disfa': au_disfa,'au_bp4d': au_bp4d}

    @property
    def stylespace_sizes(self):
        modules = list(self.input_blocks.modules()) + list(
            self.middle_block.modules()) + list(self.output_blocks.modules())
        sizes = []
        for module in modules:
            if isinstance(module, ResBlock):
                linear = module.cond_emb_layers[-1]
                sizes.append(linear.weight.shape[0])
        return sizes

    def encode_stylespace(self, x, return_vector: bool = True):
        """
        encode to style space
        """
        modules = list(self.input_blocks.modules()) + list(
            self.middle_block.modules()) + list(self.output_blocks.modules())
        # (n, c)
        cond = self.encoder.forward(x)
        S = []
        for module in modules:
            if isinstance(module, ResBlock):
                # (n, c')
                s = module.cond_emb_layers.forward(cond)
                S.append(s)

        if return_vector:
            # (n, sum_c)
            return torch.cat(S, dim=1)
        else:
            return S

    def forward(self,
                x,
                t,
                y=None,
                x_start=None,
                emo_cond=None,
                # listener_ref=None,
                # hidden_stats=None,
                cond=None,
                cond_other=None,
                return_id_feats=False,
                style=None,
                noise=None,
                t_cond=None,
                emo_label=None,
                **kwargs):
        """
        Apply the model to an input batch.

        Args:
            x_start: the original image to encode
            cond: output of the encoder
            noise: random noise (to predict the cond)
        """
        id_feats = None
        if t_cond is None:
            t_cond = t
        # pdb.set_trace()
        if noise is not None:
            # if the noise is given, we predict the cond from noise
            cond = self.noise_to_cond(noise)

        if cond is None:
            if x is not None:
                assert len(x) == len(x_start), f'{len(x)} != {len(x_start)}'
            # pdb.set_trace()
            tmp = self.encode(x_start, emo_cond=emo_cond,return_id_feats=return_id_feats)
            # if listener_ref!=None:
            #     tmp_listener = self.encode(listener_ref, return_id_feats=return_id_feats)
            # pdb.set_trace()
            # #
            # tmp['']
            # emo_label
            # 'cond': cond, 'id_listener':id_feats_l, 'id_feats': id_feats,'hidden_stats':hidden_stats,
            emotion_pred=tmp['emo_fea']#emo_fea,au_disfa,au_bp4d
            au_disfa=tmp['au_disfa']
            au_bp4d=tmp['au_bp4d']
            cond = tmp['cond']
            id_feats = tmp['id_feats']
            id_listener=tmp['id_listener']
            hidden_stats=tmp['hidden_stats']
            # pdb.set_trace()
            #simply average the condition for video slice
            # if len(cond.shape)==3:
            #     cond=torch.nn.functional.avg_pool2d(cond, (cond.shape[1], 1), stride=(cond.shape[1], 1)).squeeze()
            #     if id_feats is not None:
            #         id_feats = torch.nn.functional.avg_pool2d(id_feats, (id_feats.shape[1], 1), stride=(id_feats.shape[1], 1)).squeeze()
        else:
            # pdb.set_trace()
            emotion_pred=cond['emo_fea']#emo_fea,au_disfa,au_bp4d
            au_disfa=cond['au_disfa']
            au_bp4d=cond['au_bp4d']
            # cond = cond['cond']
            id_feats = cond['id_feats']
            id_listener=cond['id_listener']
            hidden_stats=cond['hidden_stats']
            cond=cond['cond']
        if t is not None:
            _t_emb = timestep_embedding(t, self.conf.model_channels)
            _t_cond_emb = timestep_embedding(t_cond, self.conf.model_channels)
        else:
            # this happens when training only autoenc
            _t_emb = None
            _t_cond_emb = None

        if self.conf.resnet_two_cond:
            res = self.time_embed.forward(
                time_emb=_t_emb,
                cond=cond,
                time_cond_emb=_t_cond_emb,
            )
        else:
            raise NotImplementedError()

        if self.conf.resnet_two_cond:
            # two cond: first = time emb, second = cond_emb
            emb = res.time_emb
            cond_emb = res.emb
        else:
            # one cond = combined of both time and cond
            emb = res.emb
            cond_emb = None

        # override the style if given
        style = style or res.style

        assert (y is not None) == (
            self.conf.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        if self.conf.num_classes is not None:
            raise NotImplementedError()
            # assert y.shape == (x.shape[0], )
            # emb = emb + self.label_emb(y)

        # where in the model to supply time conditions
        enc_time_emb = emb
        mid_time_emb = emb
        dec_time_emb = emb
        # where in the model to supply style conditions
        enc_cond_emb = cond_emb
        mid_cond_emb = cond_emb
        dec_cond_emb = cond_emb

        enc_cond_other_emb = None
        mid_cond_other_emb = None
        dec_cond_other_emb = None

        # hs = []
        # pdb.set_trace()
        hs = [[] for _ in range(len(self.conf.channel_mult))]
        # pdb.set_trace()
        x=F.adaptive_avg_pool3d(x,(3,128,128))
        if x is not None:
            h = x.type(self.dtype)
            # h=torch.permute(h,(0,2,1,3,4))
            h=h.permute((0,2,1,3,4))
            # input blocks
            k = 0
            for i in range(len(self.input_num_blocks)):
                for j in range(self.input_num_blocks[i]):
                    h = self.input_blocks[k](h,
                                             emb=enc_time_emb,
                                             cond=enc_cond_emb,
                                             cond_other=enc_cond_other_emb)

                    # print(i, j, h.shape)
                    hs[i].append(h)
                    k += 1
                #time downsample
                # pdb.set_trace()
                if i%2==1 and i!=5:# downsample for time dim
                # if i!=2:
                    h=torch.nn.functional.avg_pool3d(h, (2, 1, 1), stride=(2,1,1))
            assert k == len(self.input_blocks)
            # pdb.set_trace()
            # middle blocks
            h = self.middle_block(h, emb=mid_time_emb, cond=mid_cond_emb, cond_other=mid_cond_other_emb)
        else:
            # no lateral connections
            # happens when training only the autonecoder
            h = None
            hs = [[] for _ in range(len(self.conf.channel_mult))]

        # output blocks
        k = 0
        # pdb.set_trace()
        for i in range(len(self.output_num_blocks)):
            for j in range(self.output_num_blocks[i]):
                # take the lateral connection from the same layer (in reserve)
                # until there is no more, use None
                try:
                    lateral = hs[-i - 1].pop()
                    # print(i, j, lateral.shape)
                except IndexError:
                    lateral = None
                    # print(i, j, lateral)

                h = self.output_blocks[k](h,
                                          emb=dec_time_emb,
                                          cond=dec_cond_emb,
                                          lateral=lateral,
                                          cond_other=dec_cond_other_emb)
                k += 1
            #time upsample
            if i%2==1 and i!=5:# upsample for time dim
            # if i!=2:
                h = F.interpolate(h, (h.shape[2]*2, h.shape[3] , h.shape[4] ),
                                mode="nearest")
                # h=torch.nn.functional.avg_pool3d(h, (2, 1, 1), stride=(2,1,1))
        # pdb.set_trace()
        pred = self.out(h)
        
        emo_pred=self.emo_out(torch.flatten(pred[:,-1,:,:,:],start_dim=2))
        pred=pred[:,:-1,:,:,:]
        # pred=torch.permute(pred,(0,2,1,3,4))
        pred=pred.permute((0,2,1,3,4))
        pred=F.adaptive_avg_pool3d(pred,(3,256,256))
        return AutoencReturn(pred=pred, cond=cond, cond_other=cond_other, id_feats=id_feats,
            hidden_stats=hidden_stats,id_listener=id_listener,
                emotion_pred=emotion_pred,au_disfa=au_disfa,au_bp4d=au_bp4d,emo_pred=emo_pred)


class AutoencReturn(NamedTuple):
    pred:               Tensor
    cond:               Tensor = None
    cond_other:         Tensor = None
    id_feats:           Tensor = None
    emotion_pred:       Tensor =None
    au_disfa:           Tensor = None
    au_bp4d:            Tensor = None
    hidden_stats:       dict =None
    id_listener:        Tensor =None
    emo_pred:           Tensor=None


class EmbedReturn(NamedTuple):
    # style and time
    emb: Tensor = None
    # time only
    time_emb: Tensor = None
    # style only (but could depend on time)
    style: Tensor = None
    style_other: Tensor = None


class TimeStyleSeperateEmbed(nn.Module):
    # embed only style
    def __init__(self, time_channels, time_out_channels):
        super().__init__()
        self.time_embed = nn.Sequential(
            linear(time_channels, time_out_channels),
            nn.SiLU(),
            linear(time_out_channels, time_out_channels),
        )
        self.style = nn.Identity()

    def forward(self, time_emb=None, cond=None, cond_other=None, **kwargs):
        if time_emb is None:
            # happens with autoenc training mode
            time_emb = None
        else:
            time_emb = self.time_embed(time_emb)
        style = self.style(cond)
        if cond_other is not None:
            style = self.style(cond_other)
        return EmbedReturn(emb=style, time_emb=time_emb, style=style, style_other=style)

    
class DiffusionVideoAutoencoder(nn.Module):
    def __init__(self, id_file_path, lnd_file_path, bg_file_path,au_file_path_disfa=None,au_file_path_bp4d=None,emotion_file_path=None):
        super(DiffusionVideoAutoencoder, self).__init__()
        self.idnet = IDnet(id_file_path).requires_grad_(False)#don't have to modify resolution
        # self.idnet.eval()
        # self.lndnet = LNDnet(lnd_file_path).requires_grad_(False)#modify resolution
        self.bgnet = BGnet(bg_file_path).requires_grad_(False)#don't have to modify resolution
        self.linear = nn.Linear(1024, 512)
        self.lstm = nn.LSTM(512, 512, 1 , batch_first = True)
        self.emo_proj=nn.Sequential(nn.Linear(25,512),
                                    nn.Dropout(p=0.5)
                                    )
        self.emo_cond_linear=nn.Linear(512*2,512)
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
            self.emo_net.load_state_dict(torch.load(emotion_file_path, map_location='cuda'),strict=True)
            # self.emo_net.eval()
    def forward(self, id, lnd=None, emo_cond=None, return_id_feats=False,return_emo=False,expression=False):
        # pdb.set_trace()
        if emo_cond is not None:
            listener_ref=emo_cond['listener_ref']
        else:
            listener_ref=None
        hidden_stats=emo_cond['hidden_stats']
        past_emo=None#emo_cond['past_emo']
        if len(id.shape)==4:
            if lnd is None:
                lnd = id

            id_feats = self.idnet(id)  # [B, 512]
            # lnd_feats = self.lndnet(lnd) # [B, 102]
            feats = self.linear(torch.cat([id_feats, lnd_feats], dim=1))   # [B, 512]
            
            if return_id_feats:
                return feats, id_feats
            return feats
        else:  #video process
            
            
            if lnd is None:
                lnd = id
            b,v,c,w,h=id.shape
            b,v,c_l,w_l,h_l=lnd.shape

            id=id.reshape(b*v,c,w,h)
            lnd=lnd.reshape(b*v,c_l,w_l,h_l)

            # pdb.set_trace()
            self.idnet.eval()
            id_feats = self.idnet(id)  # [B, 512]

            # lnd_feats = self.lndnet(lnd) # [B, 102]
            feats = self.linear(torch.cat([id_feats, id_feats], dim=1))   # [B, 512]
            feats = feats.reshape(b,v,-1)
            id_feats=id_feats.reshape(b,v,-1)


            

            if len(feats.shape)==3:
                feats=torch.nn.functional.avg_pool2d(feats, (feats.shape[1], 1), stride=(feats.shape[1], 1)).squeeze()
                if id_feats is not None:
                    id_feats = torch.nn.functional.avg_pool2d(id_feats, (id_feats.shape[1], 1), stride=(id_feats.shape[1], 1))#.squeeze()
            # if return_emo:
            #     # pdb.set_trace()
            #     # self.emo_net.eval()
            #     emo_fea=self.emo_net(id)
            #     # emo_fea
            #     # self.au_net_DISFA.eval()
            #     # self.au_net_BP4D.eval()
            #     au_disfa=self.au_net_DISFA(id)[0]
            #     au_bp4d=self.au_net_BP4D(id)[0]

            #     return feats,emo_fea,au_disfa,au_bp4d,hidden_stats
            listener_id_feats = self.idnet(listener_ref) 
            if listener_ref != None and past_emo!=None:
                #1 get listener id
                listener_id_feats = self.idnet(listener_ref) 
                # speaker_feat=torch.cat([id_feats,au_disfa,au_bp4d,])
                # pdb.set_trace()
                #2 past emotion with lstm
                past_emo=torch.nn.functional.avg_pool2d(past_emo, (past_emo.shape[1], 1), stride=(past_emo.shape[1], 1))#.squeeze()
                # pdb.set_trace()
                listener_temporal_emo, hidden_stats = self.lstm(self.emo_proj(past_emo),hidden_stats)# (hidden_stats, hidden_stats))
                # feats=self.emo_cond_linear(
                #     torch.cat([listener_id_feats,listener_temporal_emo.squeeze(),feats],dim=-1))#fuse speaker+ listener condition
                # listener_temporal_emo, hidden_stats = self.lstm(self.emo_proj(id_feats),hidden_stats)# (hidden_stats, hidden_stats))
                feats=self.emo_cond_linear(
                    torch.cat([listener_temporal_emo.squeeze(),feats],dim=-1))#fuse speaker+ listener condition
                id_feats=id_feats.squeeze()
            return feats,hidden_stats,id_feats,listener_id_feats
            if return_id_feats:
                return feats, id_feats
            return feats
    def forward_listener(self, generated_listener,emo_cond=None,return_emo=False):
        # pdb.set_trace()
        id=generated_listener
        b,v,c,w,h=id.shape
        # b,v,c_l,w_l,h_l=lnd.shape

        id=id.view(b*v,c,w,h)
        # lnd=lnd.view(b*v,c_l,w_l,h_l)

        # pdb.set_trace()
        id_feats = self.idnet(id)  # [B, 512]
        # lnd_feats = self.lndnet(id) # [B, 102]
        # fea
        # feats = self.linear(torch.cat([id_feats, lnd_feats], dim=1))   # [B, 512]
        # feats = feats.view(b,v,-1)
        id_feats=id_feats.view(b,v,-1)


        

        # if len(feats.shape)==3:
        #     feats=torch.nn.functional.avg_pool2d(feats, (feats.shape[1], 1), stride=(feats.shape[1], 1)).squeeze()
        if id_feats is not None:
            id_feats = torch.nn.functional.avg_pool2d(id_feats, (id_feats.shape[1], 1), stride=(id_feats.shape[1], 1)).squeeze()
        # if return_emo:
        # pdb.set_trace()
        # self.emo_net.eval()
        pdb.set_trace()
        self.emo_net.eval()
        self.au_net_DISFA.eval()
        self.au_net_BP4D.eval()
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