# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained Latte.
"""
import os
import sys
try:
    import utils

    from diffusion import create_diffusion
    from download import find_model
except:
    sys.path.append(os.path.split(sys.path[0])[0])

    import utils

    from diffusion import create_diffusion
    from download import find_model

import torch
import argparse
import torchvision

from einops import rearrange
from models import get_models
from torchvision.utils import save_image
from diffusers.models import AutoencoderKL
from models.clip import TextEmbedder
import imageio
from omegaconf import OmegaConf


from datasets import get_dataset
from torch.utils.data import DataLoader
# from diffusers.models import AutoencoderKL,AutoencoderTiny
from diffusers.optimization import get_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.nn as nn
import torch.distributed as dist
from utils import (clip_grad_norm_, create_logger, update_ema, 
                   requires_grad, cleanup, create_tensorboard, 
                   write_tensorboard, setup_distributed,
                   get_experiment_dir, text_preprocessing)

import torch.nn.functional as Func
from metric import *   #import evaluation metric
import pdb
import numpy as np
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


#talking head libs
sys.path.append('/import/nobackup_mmv_ioannisp/zs003/code/face_reaction/latte_reaction/talkinghead/muse')
from talkinghead.muse.musetalk.utils.utils import get_file_type,get_video_fps,datagen
from talkinghead.muse.musetalk.utils.preprocessing import get_landmark_and_bbox,read_imgs,coord_placeholder
from talkinghead.muse.musetalk.utils.blending import get_image
from talkinghead.muse.musetalk.utils.utils import load_all_model

from transformers import pipeline
import whisper
import pdb

import edge_tts
import cv2
from tqdm import tqdm

#it is single forward function
def calculate_talkinghead_genclip(audio_file,whisper_model,audio_processor, vae, unet, pe,batch_index=0,batch_size=8,fps=25,VOICE = "en-GB-SoniaNeural"):
    # model = whisper.load_model("base.en")
    # pdb.set_trace()
    result = whisper_model.transcribe(audio_file)
    print(result["text"])
    
    sentences=result["text"].split('.')

    rand_num=torch.randint(0,100,(1,))
    for i in range(len(sentences)):
        if i==0:
            chat = [
            {"role": "system", "content": "You are a person with a specific personality, subjected to the random number:."+str(6)+". reply with 30 words"},
            {"role": "user", "content": sentences[0]}
                ]
            a=1
        else:
            chat = response[0]['generated_text']
            chat.append(
                {"role": "user", "content": sentences[i+1]}
            )
        response = pipe(chat, max_new_tokens=512)#,pad_token_id=pipe.tokenizer.eos_token_id
        current_text=response[0]['generated_text'][-1]['content']
        print(response[0]['generated_text'][-1]['content'])
        
        #text to audio
        
        # wavs = chat.infer(current_text)


        # pdb.set_trace()
        OUTPUT_audio_FILE=result_img_save_path+'/output_audio'+str(batch_index)+'.wav'
        communicate = edge_tts.Communicate(current_text, VOICE)
        communicate.save_sync(OUTPUT_audio_FILE)
        # # message in self.stream()
        wavs=[]
        # with open(OUTPUT_audio_FILE, "wb") as file:
        #     for chunk in communicate.stream_sync():
        #         if chunk["type"] == "audio":
        #             # file.write(chunk["data"])
        #             audio = np.frombuffer(chunk["data"], np.int16).astype(np.float32)*(1/32768.0)
        #             # wavs.append(chunk["data"])
        #             wavs.append(audio)
                    
        #             file.write(chunk["data"])
        #         elif chunk["type"] == "WordBoundary":
        #             print(f"WordBoundary: {chunk}")
        # communicate.save_sync(OUTPUT_FILE)
        # communicate.save_sync(OUTPUT_audio_FILE)
        # communicate.save_sync(OUTPUT_FILE)# change to talking head format

        # #audio to talking head, whisper,tiny

        #talking head
        # wavs=np.frombuffer(wavs[0], dtype=np.float16)
        # audio = np.frombuffer(data, np.int16).astype(np.float32)*(1/32768.0)
        pdb.set_trace()
        # whisper_feature = audio_processor.audio2feat(np.concatenate(wavs))
        whisper_feature = audio_processor.audio2feat(OUTPUT_audio_FILE)
        whisper_chunks = audio_processor.feature2chunks(feature_array=whisper_feature,fps=fps)

        #image processing
        # crop_frame = cv2.imread(img_path)
        #detect landmark
        
        input_img_list = [img_path, ]
        bbox_shift=0
        coord_list, frame_list = get_landmark_and_bbox(input_img_list, bbox_shift)
        # with open(crop_coord_save_path, 'wb') as f:
        #     pickle.dump(coord_list, f)
        input_latent_list = []
        for bbox, frame in zip(coord_list, frame_list):
            if bbox == coord_placeholder:
                continue
            x1, y1, x2, y2 = bbox
            # if not args.without_landmark:
            crop_frame = frame[y1:y2, x1:x2]
            # else:
            #     crop_frame = frame
            crop_frame = cv2.resize(crop_frame,(256,256),interpolation = cv2.INTER_LANCZOS4)
            latents = vae.get_latents_for_unet(crop_frame)
            input_latent_list.append(latents)
        # crop_frame=cv2.resize(crop_frame,(256,256),interpolation = cv2.INTER_LANCZOS4)
        # latents = vae.get_latents_for_unet(crop_frame)
        # input_latent_list=[]
        # input_latent_list.append(latents)
        input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
        frame_list_cycle = frame_list + frame_list[::-1]
        coord_list_cycle = coord_list + coord_list[::-1]
        video_num = len(whisper_chunks)
        # input_latent_list_cycle = [val for val in input_latent_list for i in range(len(whisper_chunks))]
        # input_latent_list_cycle= len(whisper_chunks)*input_latent_list 
        # pdb.set_trace()
        timesteps = torch.tensor([0], device=device)
        gen = datagen(whisper_chunks,input_latent_list_cycle,batch_size)
        res_frame_list=[]
        for i, (whisper_batch,latent_batch) in enumerate(tqdm(gen,total=int(np.ceil(float(video_num)/batch_size)))):
        # for whisper_batch in whisper_chunks:
            # whisper_batch=whisper_chunks
            # pdb.set_trace()
            audio_feature_batch = torch.from_numpy(whisper_batch)
            audio_feature_batch = audio_feature_batch.to(device=unet.device,
                                                            dtype=unet.model.dtype) # torch, B, 5*N,384
            audio_feature_batch = pe(audio_feature_batch)
            # latent_batch=latents
            latent_batch = latent_batch.to(dtype=unet.model.dtype)
            # pdb.set_trace()
            pred_latents = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature_batch).sample
            recon = vae.decode_latents(pred_latents)
            
            # pdb.set_trace()
            for res_frame in recon:
                res_frame_list.append(res_frame)
                # cv2.imwrite(f"{result_img_save_path}/{str(i).zfill(8)}.png",res_frame)
        print("pad talking image to original video")
        pdb.set_trace()
        for i, res_frame in enumerate(tqdm(res_frame_list)):
            bbox = coord_list_cycle[i%(len(coord_list_cycle))]
            ori_frame = copy.deepcopy(frame_list_cycle[i%(len(frame_list_cycle))])
            x1, y1, x2, y2 = bbox
            # try:
            res_frame = cv2.resize(res_frame.astype(np.uint8),(x2-x1,y2-y1))

            combine_frame = get_image(ori_frame,res_frame,bbox)
            # cv2.imwrite(f"{result_img_save_path}/{str(i).zfill(8)}.png",combine_frame)
            # pdb.set_trace()
            cv2.imwrite(f"{result_img_save_path}/{str(i).zfill(8)}.png",combine_frame)
        pdb.set_trace()
        cmd_img2video = f"ffmpeg -y -v warning -r {fps} -f image2 -i {result_img_save_path}/%08d.png -vcodec libx264 -vf format=rgb24,scale=out_color_matrix=bt709,format=yuv420p -crf 18 temp.mp4"
        print(cmd_img2video)
        os.system(cmd_img2video)
        

        output_vid_name = os.path.join('./result_talking/siyang',"siyang.mp4")
        cmd_combine_audio = f"ffmpeg -y -v warning -i {OUTPUT_audio_FILE} -i temp.mp4 {output_vid_name}"
        print(cmd_combine_audio)
        os.system(cmd_combine_audio)
        
        os.remove("temp.mp4")
        # shutil.rmtree(result_img_save_path)
        print(f"result is save to {output_vid_name}")


def calculate_talkinghead_genall(audio_file,whisper_model,audio_processor,input_latent, pipe,vae, unet, pe,batch_index=0,batch_size=8,fps=25,
        result_img_save_path='./temp',VOICE = "en-GB-SoniaNeural"):
    # model = whisper.load_model("base.en")
    pdb.set_trace()
    result = whisper_model.transcribe(audio_file)
    print(result["text"])
    
    sentences=result["text"].split('.')

    rand_num=torch.randint(0,100,(1,))
    all_text=''
    for i in range(len(sentences)):
        if i==0:
            chat = [
            {"role": "system", "content": "You are a person with a specific personality, subjected to the random number:."+str(6)+". reply with 10 words"},
            {"role": "user", "content": sentences[0]}
                ]
            a=1
        else:
            chat=response[0]['generated_text']
            chat.append(
                # {"role": "system", "content": "You are a person with a specific personality, subjected to the random number:."+str(6)+". reply with 10 words"},
                {"role": "user", "content": sentences[i]}
            )
        response = pipe(chat, max_new_tokens=512)#,pad_token_id=pipe.tokenizer.eos_token_id
        current_text=response[0]['generated_text'][-1]['content']
        all_text=all_text+current_text
        print(response[0]['generated_text'][-1]['content'])
    
    #text to audio
    
    # wavs = chat.infer(current_text)


    # pdb.set_trace()
    OUTPUT_audio_FILE=result_img_save_path+'/output_audio'+str(batch_index)+'.wav'
    communicate = edge_tts.Communicate(all_text, VOICE)
    communicate.save_sync(OUTPUT_audio_FILE)
    # # message in self.stream()
    wavs=[]
    # with open(OUTPUT_audio_FILE, "wb") as file:
    #     for chunk in communicate.stream_sync():
    #         if chunk["type"] == "audio":
    #             # file.write(chunk["data"])
    #             audio = np.frombuffer(chunk["data"], np.int16).astype(np.float32)*(1/32768.0)
    #             # wavs.append(chunk["data"])
    #             wavs.append(audio)
                
    #             file.write(chunk["data"])
    #         elif chunk["type"] == "WordBoundary":
    #             print(f"WordBoundary: {chunk}")
    # communicate.save_sync(OUTPUT_FILE)
    # communicate.save_sync(OUTPUT_audio_FILE)
    # communicate.save_sync(OUTPUT_FILE)# change to talking head format

    # #audio to talking head, whisper,tiny

    #talking head
    # wavs=np.frombuffer(wavs[0], dtype=np.float16)
    # audio = np.frombuffer(data, np.int16).astype(np.float32)*(1/32768.0)
    # pdb.set_trace()
    # whisper_feature = audio_processor.audio2feat(np.concatenate(wavs))
    whisper_feature = audio_processor.audio2feat(OUTPUT_audio_FILE)
    whisper_chunks = audio_processor.feature2chunks(feature_array=whisper_feature,fps=fps)

    # #image processing
    # # crop_frame = cv2.imread(img_path)
    # #detect landmark
    
    # input_img_list = [img_path, ]
    # bbox_shift=0
    # coord_list, frame_list = get_landmark_and_bbox(input_img_list, bbox_shift)
    # # with open(crop_coord_save_path, 'wb') as f:
    # #     pickle.dump(coord_list, f)
    # input_latent_list = []
    # for bbox, frame in zip(coord_list, frame_list):
    #     if bbox == coord_placeholder:
    #         continue
    #     x1, y1, x2, y2 = bbox
    #     # if not args.without_landmark:
    #     crop_frame = frame[y1:y2, x1:x2]
    #     # else:
    #     #     crop_frame = frame
    #     crop_frame = cv2.resize(crop_frame,(256,256),interpolation = cv2.INTER_LANCZOS4)
    #     latents = vae.get_latents_for_unet(crop_frame)
    #     input_latent_list.append(latents)
    # # crop_frame=cv2.resize(crop_frame,(256,256),interpolation = cv2.INTER_LANCZOS4)
    # # latents = vae.get_latents_for_unet(crop_frame)
    # # input_latent_list=[]
    # # input_latent_list.append(latents)
    pdb.set_trace()
    input_latent_list=[]
    for single_frame in input_latent:
        # crop_frame=cv2.resize(single_frame,(256,256),interpolation = cv2.INTER_LANCZOS4)
        latents = vae.get_latents_for_unet_nonorm(single_frame)
        input_latent_list.append(latents)

    #input latent start, directly use latent
    # input_latent_list=input_latent
    # input_latent_list_cycle=torch.cat((input_latent_list,torch.flip(input_latent_list,[0])),dim=0)

    #----
    #use image----
    input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
    
    # frame_list_cycle = frame_list + frame_list[::-1]
    # coord_list_cycle = coord_list + coord_list[::-1]
    video_num = len(whisper_chunks)
    # input_latent_list_cycle = [val for val in input_latent_list for i in range(len(whisper_chunks))]
    # input_latent_list_cycle= len(whisper_chunks)*input_latent_list 
    # pdb.set_trace()
    timesteps = torch.tensor([0], device=unet.device)
    gen = datagen(whisper_chunks,input_latent_list_cycle,batch_size)
    res_frame_list=[]
    res_frame_list_latent=[]
    for i, (whisper_batch,latent_batch) in enumerate(tqdm(gen,total=int(np.ceil(float(video_num)/batch_size)))):
    # for whisper_batch in whisper_chunks:
        # whisper_batch=whisper_chunks
        # pdb.set_trace()
        audio_feature_batch = torch.from_numpy(whisper_batch)
        audio_feature_batch = audio_feature_batch.to(device=unet.device,
                                                        dtype=unet.model.dtype) # torch, B, 5*N,384
        audio_feature_batch = pe(audio_feature_batch)
        # latent_batch=latents
        latent_batch = latent_batch.to(dtype=unet.model.dtype)
        # pdb.set_trace()
        pred_latents = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature_batch).sample
        #mask talent


        #do not reconstruct images if use reaction framework
        # recon = vae.decode_latents(pred_latents)
        
        # pdb.set_trace()
        # for res_frame in recon:
        #     res_frame_list.append(res_frame)
        for res_latent in pred_latents:
            #mask latent
            # res_latent_detrans=res_latent*0.5+0.5
            # res_latent_detrans = res_latent_detrans * (self._mask_tensor>0.5)
            res_frame_list_latent.append(res_latent)
            # cv2.imwrite(f"{result_img_save_path}/{str(i).zfill(8)}.png",res_frame)
    # print("pad talking image to original video")
    # pdb.set_trace()
    # for i, res_frame in enumerate(tqdm(res_frame_list)):
    #     bbox = coord_list_cycle[i%(len(coord_list_cycle))]
    #     ori_frame = copy.deepcopy(frame_list_cycle[i%(len(frame_list_cycle))])
    #     x1, y1, x2, y2 = bbox
    #     # try:
    #     res_frame = cv2.resize(res_frame.astype(np.uint8),(x2-x1,y2-y1))

    #     combine_frame = get_image(ori_frame,res_frame,bbox)
    #     # cv2.imwrite(f"{result_img_save_path}/{str(i).zfill(8)}.png",combine_frame)
    #     # pdb.set_trace()
    #     cv2.imwrite(f"{result_img_save_path}/{str(i).zfill(8)}.png",combine_frame)
    # pdb.set_trace()
    # cmd_img2video = f"ffmpeg -y -v warning -r {fps} -f image2 -i {result_img_save_path}/%08d.png -vcodec libx264 -vf format=rgb24,scale=out_color_matrix=bt709,format=yuv420p -crf 18 temp.mp4"
    # print(cmd_img2video)
    # os.system(cmd_img2video)
    

    # output_vid_name = os.path.join('./result_talking/siyang',"siyang.mp4")
    # cmd_combine_audio = f"ffmpeg -y -v warning -i {OUTPUT_audio_FILE} -i temp.mp4 {output_vid_name}"
    # print(cmd_combine_audio)
    # os.system(cmd_combine_audio)
    
    # os.remove("temp.mp4")
    # # shutil.rmtree(result_img_save_path)
    # print(f"result is save to {output_vid_name}")        
def main(args):
    # Setup PyTorch:
    # torch.manual_seed(args.seed)
    setup_distributed()
    # dist.init_process_group("nccl")
    # assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    # rank = dist.get_rank()
    # device = rank % torch.cuda.device_count()
    # local_rank = rank

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    # device = torch.device("cuda", 0)
    clip_size=32 #64
    audio_file='/import/nobackup_mmv_ioannisp/zs003/face_react/new_audio_folder/all_together/NoXI/026_2016-04-06_Nottingham/Expert_video/3.wav'

    #
    # pdb.set_trace()
    print('check test/val matrix..')
    # if val_test == 'val':
    args.dataset_path='/import/nobackup_mmv_ioannisp/zs003/code/face_reaction/latte_reaction/val_test_matrix'
    speaker_neighbour_matrix = np.load(os.path.join(args.dataset_path, 'neighbour_emotion_val.npy'))
    # else:
    speaker_neighbour_matrix = np.load(os.path.join(args.dataset_path, 'neighbour_emotion_test.npy'))

    # seed = args.global_seed + rank
    # torch.manual_seed(seed)
    torch.cuda.set_device(device)
    # print(f"Starting rank={rank}, local rank={local_rank}, seed={seed}, world_size={dist.get_world_size()}.")


    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"

    if args.ckpt is None:
        assert args.model == "Latte-XL/2", "Only Latte-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    #load audio->talking head branch--------------------------------
    whisper_model = whisper.load_model("base.en")

    # result = whisper_model.transcribe(audio_file)
    # print(result["text"])

    # sentences=result["text"].split('.')
    ##LLM processing
    # pipe = pipeline("text-generation", "meta-llama/Meta-Llama-3-8B-Instruct", torch_dtype=torch.bfloat16, device_map="auto")
    # pipe = pipeline("text-generation", "meta-llama/Meta-Llama-3-8B-Instruct", torch_dtype=torch.float16, device_map="auto")
    pipe = pipeline("text-generation", "meta-llama/Meta-Llama-3-8B-Instruct", torch_dtype=torch.float16, device=0)

    #initlize chat
    
    #init talking head
    
    # load model weights
    # pdb.set_trace()
    audio_processor, vae, unet, pe = load_all_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timesteps = torch.tensor([0], device=device)


    #end audio->talking head----------------------------------------
    using_cfg = args.cfg_scale > 1.0

    # Load model:
    latent_size = args.image_size // 8
    args.latent_size = latent_size
    model = get_models(args).to(device)

    if args.use_compile:
        model = torch.compile(model)

    # a pre-trained model or load a custom Latte checkpoint from train.py:
    ckpt_path = args.ckpt
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)

    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    # vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae").to(device)
    # vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae").to(device)
    # text_encoder = TextEmbedder().to(device)

    if args.use_fp16:
        print('WARNING: using half percision for inferencing!')
        vae.to(dtype=torch.float16)
        model.to(dtype=torch.float16)
        # text_encoder.to(dtype=torch.float16)
    dataset = get_dataset(args,train_mode='test')


    
    sampler = DistributedSampler(
    dataset,
    num_replicas=dist.get_world_size(),
    rank=0,
    shuffle=True,
    seed=args.global_seed
    )
    batch_size=int(args.local_batch_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    listener_emotion_pred_list=[]
    listener_emotion_gt_list=[]
    speaker_emotion_gt_list=[]
    all_listener_emotion_pred_list=[]
    # pdb.set_trace()
    # Labels to condition the model with (feel free to change):
    for step, video_data in enumerate(loader):
        # Skip steps until we reach the resumed step
        # if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
        #     continue
        out_samples=None
        # pdb.set_trace()
        for index_clip in range((video_data['img'].shape)[1]//clip_size):
            x = video_data['img'].to(device, non_blocking=True)[:,index_clip*clip_size:(index_clip+1)*clip_size]#.half()
            listener_ref=video_data['listener_video_clip'].to(device, non_blocking=True)[:,index_clip*clip_size:(index_clip+1)*clip_size]#.half()
            emo_label=video_data['listener_emotion'].to(device, non_blocking=True)[:,index_clip*clip_size:(index_clip+1)*clip_size]
            speaker_emo=video_data['speaker_emotion'].to(device, non_blocking=True)[:,index_clip*clip_size:(index_clip+1)*clip_size]
            original_x=x
            # listener_ref=listener_ref
            # Create sampling noise:
            if args.use_fp16:
                z = torch.randn(batch_size, args.num_frames, 4, latent_size, latent_size, dtype=torch.float16, device=device) # b c f h w
            else:
                z = torch.randn(batch_size, args.num_frames, 4, latent_size, latent_size, device=device)

            # Setup classifier-free guidance:
            # z = torch.cat([z, z], 0)
            if using_cfg:
                z = torch.cat([z, z], 0)
                y = torch.randint(0, args.num_classes, (1,), device=device)
                y_null = torch.tensor([101] * 1, device=device)
                y = torch.cat([y, y_null], dim=0)
                model_kwargs = dict(y=y, cfg_scale=args.cfg_scale, use_fp16=args.use_fp16)
                sample_fn = model.forward_with_cfg
            else:
                sample_fn = model.forward
                model_kwargs = dict(y=None, use_fp16=args.use_fp16)
            # pdb.set_trace()
            # Sample images:
            if args.sample_method == 'ddim':
                samples = diffusion.ddim_sample_loop(
                    sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device,
                    original_x=x,listener_ref=listener_ref,
                )
            elif args.sample_method == 'ddpm':
                samples = diffusion.p_sample_loop(
                    sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device,
                    original_x=x,listener_ref=listener_ref,
                )
            # pdb.set_trace()
            print(samples.shape)
            if args.use_fp16:
                samples = samples.to(dtype=torch.float16)
            b, f, c, h, w = samples.shape
            # pdb.set_trace()
            samples = rearrange(samples, 'b f c h w -> (b f) c h w')
            # if index_clip==0:
            #     out_samples=rearrange(samples, '(b f) c h w -> b f c h w', b=b)
            # else:
            #     out_samples=torch.cat((out_samples,rearrange(samples, '(b f) c h w -> b f c h w', b=b)),dim=1)
            samples = vae.decode(samples / 0.18215).sample
            samples = rearrange(samples, '(b f) c h w -> b f c h w', b=b)
            if index_clip==0:
                out_samples=samples#rearrange(samples, '(b f) c h w -> b f c h w', b=b)
            else:
                out_samples=torch.cat((out_samples,samples),dim=1)
            
        pdb.set_trace()
        #can consider do not use the decode of vae
        for batch_index in range(len(out_samples)):

            calculate_talkinghead_genall(audio_file,whisper_model,audio_processor,out_samples[batch_index], pipe,vae, unet, pe)
            pdb.set_trace()
        # #generate talking head
        # result = whisper_model.transcribe(audio_file)
        # print(result["text"])

        # sentences=result["text"].split('.')




        #visualize 
        video_ = ((video_data['img'][0] * 0.5 + 0.5) * 255).add_(0.5).clamp_(0, 255).to(dtype=torch.uint8).cpu().permute(0, 2, 3, 1).contiguous()
        video_save_path = os.path.join(args.save_video_path, 'original' + str(step)+'.mp4')
        print(video_save_path)
        imageio.mimwrite(video_save_path, video_, fps=8, quality=9)
        print('save path {}'.format(args.save_video_path))

        video_ = ((out_samples[0] * 0.5 + 0.5) * 255).add_(0.5).clamp_(0, 255).to(dtype=torch.uint8).cpu().permute(0, 2, 3, 1).contiguous()
        video_save_path = os.path.join(args.save_video_path, 'sample_test_ddpm' + str(step)+'.mp4')
        print(video_save_path)
        imageio.mimwrite(video_save_path, video_, fps=8, quality=9)
        print('save path {}'.format(args.save_video_path))
        pdb.set_trace()
        # Save and display images:
        # pdb.set_trace()

        #get emo feature
        id_feats,emo_fea,au_disfa,au_bp4d=model.facial_condition.forward_listener(samples)
        # id=id.reshape(b*v,c,w,h)
        # lnd=lnd.reshape(b*v,c_l,w_l,h_l)
        
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
        current_emo=torch.cat([emo,au],dim=-1)

        #
        # pdb.set_trace()
        listener_emotion_pred_list.append(current_emo.cpu())
        listener_emotion_gt_list.append(emo_label.cpu())
        speaker_emotion_gt_list.append(speaker_emo)
        

    pdb.set_trace()
    listener_emotion_pred = torch.cat(listener_emotion_pred_list, dim = 0)
    listener_emotion_gt = torch.cat(listener_emotion_gt_list, dim = 0)

    speaker_emotion_gt=torch.cat(speaker_emotion_gt_list, dim = 0)

    all_listener_emotion_pred_list.append(listener_emotion_pred.unsqueeze(1))
    all_listener_emotion_pred = torch.cat(all_listener_emotion_pred_list, dim=1)
    # if not os.path.exists(args.save_video_path):
    #     os.makedirs(args.save_video_path)

    print("-----------------Evaluating Metric-----------------")

    # p = args.threads
    
    # If you have problems running function compute_TLCC_mp, please replace this function with function compute_TLCC
    TLCC = compute_TLCC(all_listener_emotion_pred.cpu(), speaker_emotion_gt.cpu())
    
    # If you have problems running function compute_FRC_mp, please replace this function with function compute_FRC
    FRC = compute_FRC(args, all_listener_emotion_pred.cpu(), listener_emotion_gt.cpu(), val_test='test')

    # If you have problems running function compute_FRD_mp, please replace this function with function compute_FRD
    FRD = compute_FRD(args, all_listener_emotion_pred.cpu(), listener_emotion_gt.cpu(), val_test='test')

    FRDvs = compute_FRDvs(all_listener_emotion_pred.cpu())
    FRVar  = compute_FRVar(all_listener_emotion_pred.cpu())
    smse  = compute_s_mse(all_listener_emotion_pred.cpu())
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/reaction/reaction_test.yaml")
    # parser.add_argument("--ckpt", type=str, default="/data/scratch/acw569/checkpoint/latte_check/001-Latte-S-2-F64S3-reaction-Amp/checkpoints/0130000.pt")
    parser.add_argument("--ckpt", type=str, default="/import/nobackup_mmv_ioannisp/zs003/code/face_reaction/latte_reaction/results/120-Latte-S-2-F32S3-reaction-Amp/checkpoints/0080000.pt")

    
    #wrong label:/import/nobackup_mmv_ioannisp/zs003/code/face_reaction/latte_reaction/results/092-Latte-S-2-F32S3-reaction-Amp/checkpoints/0240000.pt
    parser.add_argument("--save_video_path", type=str, default="./sample_videos/")
    args = parser.parse_args()
    omega_conf = OmegaConf.load(args.config)
    omega_conf.ckpt = args.ckpt
    omega_conf.save_video_path = args.save_video_path
    main(omega_conf)
