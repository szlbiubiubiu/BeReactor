import os
import cv2
import numpy as np
import torch

ffmpeg_path = os.getenv('FFMPEG_PATH')
if ffmpeg_path is None:
    print("please download ffmpeg-static and export to FFMPEG_PATH. \nFor example: export FFMPEG_PATH=/musetalk/ffmpeg-4.4-amd64-static")
elif ffmpeg_path not in os.getenv('PATH'):
    print("add ffmpeg to path")
    os.environ["PATH"] = f"{ffmpeg_path}:{os.environ['PATH']}"

    
from talkinghead.muse.musetalk.whisper.audio2feature import Audio2Feature
from talkinghead.muse.musetalk.models.vae import VAE
from talkinghead.muse.musetalk.models.unet import UNet,PositionalEncoding

import pdb
def load_all_model():
    # audio_processor = Audio2Feature(model_path="./models/whisper/tiny.pt")
    # vae = VAE(model_path = "./models/sd-vae-ft-mse/")
    # unet = UNet(unet_config="./models/musetalk/musetalk.json",
    #             model_path ="./models/musetalk/pytorch_model.bin")
    # pdb.set_trace()
    audio_processor = Audio2Feature(model_path="/import/nobackup_mmv_ioannisp/zs003/code/face_reaction/latte_reaction/talkinghead/muse/models/whisper/tiny.pt")
    vae = VAE(model_path = "/import/nobackup_mmv_ioannisp/zs003/code/face_reaction/latte_reaction/talkinghead/muse/models/sd-vae-ft-mse/")
    unet = UNet(unet_config="/import/nobackup_mmv_ioannisp/zs003/code/face_reaction/latte_reaction/talkinghead/muse/models/musetalk/musetalk.json",
                model_path ="/import/nobackup_mmv_ioannisp/zs003/code/face_reaction/latte_reaction/talkinghead/muse/models/musetalk/pytorch_model.bin")
                
    pe = PositionalEncoding(d_model=384)
    return audio_processor,vae,unet,pe

def get_file_type(video_path):
    _, ext = os.path.splitext(video_path)

    if ext.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
        return 'image'
    elif ext.lower() in ['.avi', '.mp4', '.mov', '.flv', '.mkv']:
        return 'video'
    else:
        return 'unsupported'

def get_video_fps(video_path):
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    video.release()
    return fps

def datagen(whisper_chunks,
            vae_encode_latents,
            batch_size=8,
            delay_frame=0):
    whisper_batch, latent_batch = [], []
    for i, w in enumerate(whisper_chunks):
        idx = (i+delay_frame)%len(vae_encode_latents)
        latent = vae_encode_latents[idx]
        whisper_batch.append(w)
        latent_batch.append(latent)

        if len(latent_batch) >= batch_size:
            whisper_batch = np.stack(whisper_batch)
            latent_batch = torch.cat(latent_batch, dim=0)
            yield whisper_batch, latent_batch
            whisper_batch, latent_batch = [], []

    # the last batch may smaller than batch size
    if len(latent_batch) > 0:
        whisper_batch = np.stack(whisper_batch)
        latent_batch = torch.cat(latent_batch, dim=0)

        yield whisper_batch, latent_batch

def datagen_reaction(whisper_chunks,
            vae_encode_latents,
            batch_size=8,
            delay_frame=0):
    whisper_batch, latent_batch = [], []
    laten_repeat=len(whisper_chunks)//len(vae_encode_latents)
    pdb.set_trace()
    vae_encode_latents_new=[]
    for single_latent in vae_encode_latents:
        repeated_latent=laten_repeat * [single_latent]
        vae_encode_latents_new+=repeated_latent
    vae_encode_latents=vae_encode_latents_new
    pdb.set_trace()
    # vae_encode_latents_new=[laten_repeat * [single_latent] for single_latent in vae_encode_latents]
    ##expand latent, repeat chanel-->stack on image
    # vae_encode_latents=einops.repeat(vae_encode_latents, 'b c h w ->  b (laten_repeat c)h w', repeat=laten_repeat)
    for i, w in enumerate(whisper_chunks):
        idx = (i+delay_frame)%len(vae_encode_latents)
        latent = vae_encode_latents[idx]
        whisper_batch.append(w)
        latent_batch.append(latent)

        if len(latent_batch) >= batch_size:
            whisper_batch = np.stack(whisper_batch)
            latent_batch = torch.cat(latent_batch, dim=0)
            yield whisper_batch, latent_batch
            whisper_batch, latent_batch = [], []

    # the last batch may smaller than batch size
    if len(latent_batch) > 0:
        whisper_batch = np.stack(whisper_batch)
        latent_batch = torch.cat(latent_batch, dim=0)

        yield whisper_batch, latent_batch
