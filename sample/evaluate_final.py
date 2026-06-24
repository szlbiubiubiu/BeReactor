

"""
In this file, please ignore the pdb
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


from models.model_irse import IDnet
from models.basenet import LNDnet
from models.seg_model_2 import BGnet
from models.emo_net import EmoNet
from models.emo_fr_net.au_model import MEFL
import torch.nn.functional as Func
from metric import *   #import evaluation metric
from metric.react2025_me.FRC import compute_FRC_nomp
from metric.react2025_me.FRD import compute_FRD_nomp
from metric.react2025_me.FRDvs import compute_FRDvs as compute_FRDvs_2025
from metric.react2025_me.FRVar import compute_FRVar as compute_FRVar_2025
from metric.react2025_me.TLCC import compute_TLCC_nomp
import numpy as np
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

os.environ["TOKENIZERS_PARALLELISM"] = "false"
#talking head libs
# sys.path.append('/import/nobackup_mmv_ioannisp/zs003/code/face_reaction/latte_reaction/talkinghead/muse')
from talkinghead.muse.musetalk.utils.utils import get_file_type,get_video_fps,datagen,datagen_reaction
from talkinghead.muse.musetalk.utils.preprocessing import get_landmark_and_bbox,read_imgs,coord_placeholder
from talkinghead.muse.musetalk.utils.blending import get_image
from talkinghead.muse.musetalk.utils.utils import load_all_model

from transformers import pipeline, AutoTokenizer
import whisper

import edge_tts
import soundfile as sf
from parler_tts import ParlerTTSForConditionalGeneration
import cv2
from tqdm import tqdm
import copy
import torch.nn.functional as F
import re
import json as _json
from decord import VideoReader, cpu as decord_cpu
from PIL import Image
from torchvision import transforms as T
import glob
import platform
import time
#it is single forward function
import shutil
import pdb


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
        # pdb.set_trace()
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
        # pdb.set_trace()
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
        # pdb.set_trace()
        cmd_img2video = f"ffmpeg -y -v warning -r {fps} -f image2 -i {result_img_save_path}/%08d.png -vcodec libx264 -vf format=rgb24,scale=out_color_matrix=bt709,format=yuv420p -crf 18 temp.mp4"
        print(cmd_img2video)
        os.system(cmd_img2video)
        

        output_vid_name = os.path.join('./result_talking/siyang',"siyang.mp4")
        cmd_combine_audio = f"ffmpeg -y -v warning -i {OUTPUT_audio_FILE} -i temp.mp4 {output_vid_name}"
        print(cmd_combine_audio)
        os.system(cmd_combine_audio)
        
        os.remove("temp.mp4")
        # Remove PNG image files
        os.system(f"rm {result_img_save_path}/*.png")
        # shutil.rmtree(result_img_save_path)
        print(f"result is save to {output_vid_name}")

def chat_with_llama(model, tokenizer, conversation_history):
        # Tokenize the input along with the conversation history
        inputs = tokenizer(conversation_history, return_tensors="pt")
        outputs = model.generate(**inputs, max_length=1000, do_sample=True)
        
        # Decode the response
        response = tokenizer.decode(outputs[:, inputs['input_ids'].shape[-1]:][0], skip_special_tokens=True)
        return response

# def forward_listener():


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
        # pdb.set_trace()
        if au_file_path_disfa is not None:
            # pdb.set_trace()
            self.au_net_DISFA=MEFL.MEFARG(num_classes=8).requires_grad_(False)# resnet
            self.au_net_DISFA.load_state_dict(torch.load(au_file_path_disfa, map_location='cuda')['state_dict'])
            # self.au_net_DISFA.eval()
            self.au_net_BP4D=MEFL.MEFARG(num_classes=12).requires_grad_(False) #num_classes: 12   neighbor_num: 4
            self.au_net_BP4D.load_state_dict(torch.load(au_file_path_bp4d, map_location='cuda')['state_dict'])
            # self.au_net_BP4D.eval()
        if emotion_file_path is not None:
            #input resolution is 256
            self.emo_net=EmoNet(n_expression=8).requires_grad_(False)#what is the resolution#.to(device)()
            self.emo_net.load_state_dict(torch.load(emotion_file_path, map_location='cuda'),strict=True)
            # self.emo_net.eval()
    def forward_listener(self, generated_listener,clip_leng=32,emo_cond=None,return_emo=False):
        # pdb.set_trace()
        

        id=generated_listener
        if len(generated_listener.shape)==5:
            b,v,c,w,h=id.shape
            # b,v,c_l,w_l,h_l=lnd.shape

            id=id.reshape(b*v,c,w,h)
        elif len(generated_listener.shape)==4:
            b,c,w,h=id.shape
        # b,v,c_l,w_l,h_l=lnd.shape

        # id=id.view(b*v,c,w,h)
        # lnd=lnd.view(b*v,c_l,w_l,h_l)

        # pdb.set_trace()
        id_feats = self.idnet(id)  # [B, 512]
        # lnd_feats = self.lndnet(id) # [B, 102]
        # fea
        # feats = self.linear(torch.cat([id_feats, lnd_feats], dim=1))   # [B, 512]
        # feats = feats.view(b,v,-1)
        id_feats=id_feats.reshape(b//clip_leng,clip_leng,-1)


        

        # if len(feats.shape)==3:
        #     feats=torch.nn.functional.avg_pool2d(feats, (feats.shape[1], 1), stride=(feats.shape[1], 1)).squeeze()
        if id_feats is not None:
            id_feats = torch.nn.functional.avg_pool2d(id_feats, (id_feats.shape[1], 1), stride=(id_feats.shape[1], 1)).squeeze()
        
        self.emo_net.eval()
        # self.au_net_DISFA.eval()
        # self.au_net_BP4D.eval()
        emo_fea=self.emo_net(id)
        
        # # expression_soft=F.softmax(emo_fea,dim=1)
        # # emo_fea
        # self.au_net_DISFA.eval()
        # self.au_net_BP4D.eval()
        # au_disfa=self.au_net_DISFA(id)[0]
        # au_bp4d=self.au_net_BP4D(id)[0]

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

pretrain_root = ''
encoder_emo=DiffusionVideoAutoencoder(pretrain_root+"./pretrained_models/model_ir_se50.pth", pretrain_root+"./pretrained_models/mobilenet_224_model_best_gdconv_external.pth.tar", pretrain_root+"./pretrained_models/79999_iter.pth"
                    ,emotion_file_path=pretrain_root+'./pretrained_models/emonet_8.pth', 
                    au_file_path_disfa=pretrain_root+'./pretrained_models/MEFARG_swin_base_DISFA_fold1.pth',
                    au_file_path_bp4d=pretrain_root+'./pretrained_models/MEFARG_swin_base_BP4D_fold1.pth').to(torch.device('cuda:0'))

# Initialize UTMOS model for audio quality evaluation
try:
    import utmosv2
    import librosa
    utmos_model = utmosv2.create_model(pretrained=False)
    utmos_model.load_state_dict(torch.load('./fold0_s42_best_model.pth'))
    utmos_model = utmos_model.to(torch.device('cuda:0'))
    utmos_model.eval()
    utmos_available = True
except Exception as e:
    print(f"UTMOS model initialization failed: {e}")
    utmos_available = False

# Initialize ParlerTTS model for timeline-aware audio generation
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
audio_device = "cuda:1" if torch.cuda.device_count() > 1 else ("cuda:0" if torch.cuda.is_available() else "cpu")
audio_model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-v1").to(audio_device)
tts_tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")

from openai import OpenAI
import openai

import nltk
# nltk.download('punkt_tab')
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import bert_score

# Function to calculate BLEU for multiple chats
def calculate_bleu_batch(references, candidates):
    smoothie = SmoothingFunction().method4
    bleu_scores = []
    for ref, cand in zip(references, candidates):
        score = sentence_bleu([nltk.word_tokenize(ref)], nltk.word_tokenize(cand), smoothing_function=smoothie)
        bleu_scores.append(score)
    return bleu_scores

# Function to calculate ROUGE for multiple chats
def calculate_rouge_batch(references, candidates):
    # scorer = rouge_scorer.RougeScorer(['rouge-1', 'rouge-2', 'rouge-l'], use_stemmer=True)
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_list = []
    for ref, cand in zip(references, candidates):
        scores = scorer.score(ref, cand)
        rouge_list.append(scores)
    return rouge_list

# Function to calculate BERTScore for multiple chats
def calculate_bertscore_batch(references, candidates):
    P, R, F1 = bert_score.score(candidates, references, lang="en")
    return F1.tolist()  # Return list of F1 scores for each pair

# def eva_METEOR(dialogue, listener_audio):
#     reference=dialogue
def evaluate_LLM(dialogue: str):
    """
    Evaluates the realism of a dialogue using an LLM with chain-of-thought prompting.
    
    Args:
        dialogue (str): The dialogue text to evaluate.
    
    Returns:
        dict: A dictionary containing the chain-of-thought reasoning and final realism score.
    """
    # Construct a prompt with clear instructions and examples.
    prompt = (
        "You are a dialogue evaluation expert. Your task is to assess the realism of a given dialogue. "
        "Realism includes factors such as coherence, natural language usage, contextual relevance, and emotional nuance. "
        "Please follow these steps:\n\n"
        "1. Analyze the dialogue text and explain step-by-step your reasoning regarding its realism. "
        "Consider the flow, tone, and whether it sounds like a human conversation.\n"
        "2. Based on your analysis, provide a final realism score on a scale from 1 (not realistic) to 5 (highly realistic).\n\n"
        "3. Based on your analysis, provide a sentiment analysis by choosing number from 'Neutral(0), Happy(1), Sad(2), Surprise(3), Fear(4), Disgust(5), Anger(6), Contempt(7)', and give valence and arousal score(both range from 0-1).\n"
        "Dialogue:\n"
        f"'''{dialogue}'''\n\n"
        "Please output your answer in JSON format as follows:\n"
        "{\n"
        '  "chain_of_thought": "Your step-by-step reasoning here...",\n'
        '  "realism_score": X\n'
        '  "sentiment analysis": X\n'
        '  "valence_score": X\n'
        '  "arousal_score": X\n'
        "}\n"
    )
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        base_url=os.environ.get("OPENAI_BASE_URL", "https://api.chatanywhere.tech/v1"),
    )
    
    response = client.chat.completions.create(
            # model="gpt-3.5-turbo",  # or your preferred model
            model="gpt-4o-mini-ca",
            # model="gpt-3.5-turbo-ca",
            
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,  # lower temperature for more deterministic responses
        max_tokens=500,
        n=1,
        stop=None,
        )
    # # Call the ChatCompletion endpoint with a chain-of-thought prompt
    # response = openai.ChatCompletion.create(
    #     model="gpt-4",  # or your preferred model
    #     messages=[
    #         {"role": "system", "content": "You are a helpful assistant."},
    #         {"role": "user", "content": prompt}
    #     ],
    #     temperature=0.2,  # lower temperature for more deterministic responses
    #     max_tokens=500,
    #     n=1,
    #     stop=None,
    # )
    # pdb.set_trace()
    # Parse the response content (assuming the output is valid JSON)
    answer = response.choices[0].message.content.strip()
    # all_scores= re.findall(r'-?\d+\.?\d*', answer)
    # converted_numbers = []
    # for num_str in all_scores:
    #     try:
    #         # 优先尝试转换为 int（适合没有小数点的数字）
    #         converted_numbers.append(int(num_str))
    #     except ValueError:
    #         # 如果转 int 失败（说明是小数），则转换为 float
    #         converted_numbers.append(float(num_str))
    # arousal_score=converted_numbers[-1]
    # valence_score=converted_numbers[-2]
    # sentiment_score=converted_numbers[-3]
    # realism_score=converted_numbers[-4]

    arousal_score = re.findall(r'\d+\.\d+', answer.split('\n')[-2])
    valence_score = re.findall(r'\d+\.\d+', answer.split('\n')[-3])
    sentiment_score = re.findall(r'\d+', answer.split('\n')[-4])
    realism_score = re.findall(r'\d+', answer.split('\n')[-5])
    # numbers = re.findall(r'\d+', answer.split('\n')[-2])
    arousal_score = [float(num) for num in arousal_score]
    valence_score = [float(num) for num in valence_score]
    sentiment_score = [int(num) for num in sentiment_score]
    
    realism_score = [int(num) for num in realism_score]
    if len(arousal_score)==0:
        arousal_score=0
        valence_score=0
        sentiment_score=0
        realism_score=0
    elif len(realism_score)==1:
        # pdb.set_trace()
        arousal_score=arousal_score[0]
        valence_score=valence_score[0]
        sentiment_score=sentiment_score[0]
        realism_score=realism_score[0]




    # arousal_score = re.findall(r'\d+\.\d+', answer.split('\n')[-1])
    # valence_score = re.findall(r'\d+\.\d+', answer.split('\n')[-2])
    # sentiment_score = re.findall(r'\d+', answer.split('\n')[-3])
    # realism_score = re.findall(r'\d+', answer.split('\n')[-4])
    # # numbers = re.findall(r'\d+', answer.split('\n')[-2])
    # pdb.set_trace()
    # arousal_score = [float(num) for num in arousal_score]
    # valence_score = [float(num) for num in valence_score]
    # sentiment_score = [int(num) for num in sentiment_score]
    
    # realism_score = [int(num) for num in realism_score]
    
    
    # # Process arousal_score
    # if len(arousal_score) == 0:
    #     arousal_score = 0
    # elif len(arousal_score) == 1:
    #     arousal_score = arousal_score[0]
    
    # # Process valence_score
    # if len(valence_score) == 0:
    #     valence_score = 0
    # elif len(valence_score) == 1:
    #     valence_score = valence_score[0]
    
    # # Process sentiment_score
    # if len(sentiment_score) == 0:
    #     sentiment_score = 0
    # elif len(sentiment_score) == 1:
    #     sentiment_score = sentiment_score[0]
    
    # # Process realism_score
    # if len(realism_score) == 0:
    #     realism_score = 0
    # elif len(realism_score) == 1:
    #     realism_score = realism_score[0]
        
    # # pdb.set_trace()
    # if sentiment_score>7:
    # pdb.set_trace()
        # a=1
    sentiment_emb=F.one_hot(torch.tensor(sentiment_score),num_classes=10)
    sentiment_emb[-2]=valence_score
    sentiment_emb[-1]=arousal_score
    sentiment_emb=torch.unsqueeze(sentiment_emb,dim=0)
    # try:
    #     import json
    #     result = json.loads(answer)
    # except json.JSONDecodeError:
    #     # If JSON parsing fails, return the raw answer
    #     result = {"raw_output": answer}
    
    return realism_score,sentiment_emb#sentiment_score,valence_score,arousal_score

def load_frames_from_video_path(video_path, start_frame, num_frames, img_size=256, stride=1):
    """Read *num_frames* frames starting at *start_frame* from a video file.

    Returns ``(clip, total_length)`` where *clip* is a float32 tensor of shape
    ``[1, num_frames, 3, img_size, img_size]`` in ``[-1, 1]`` and
    *total_length* is the total number of frames in the source video.
    """
    with open(video_path, 'rb') as f:
        vr = VideoReader(f, ctx=decord_cpu(0))
    total_length = len(vr)
    _tf = T.Compose([T.Resize(img_size), T.ToTensor(),
                     T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    clip = []
    for i in range(num_frames):
        src_idx = min(start_frame + i * stride, total_length - 1)
        frame = vr[src_idx]
        img = Image.fromarray(frame.asnumpy())
        clip.append(_tf(img).unsqueeze(0))
    clip = torch.cat(clip, dim=0).unsqueeze(0)  # [1, num_frames, 3, H, W]
    return clip, total_length


def openai_deepseek_talkinghead_genall(audio_file, listener_audio_file, whisper_model, audio_processor,
    speaker_audio_path, listener_audio_path, emo_label, input_latent, pipe, vae, unet, pe,
    batch_index=0, batch_size=8, fps=10,
    result_img_save_path='./temp', VOICE="en-GB-SoniaNeural",
    return_image4emotion=True, num_frame_sp=32,
    segment_list=None, listener_video_path=None, listener_num_frames=None,
    tts_method='edge'):
    # frames_per_seg: derived from data; 64 is the standard clip length
    n_segs_hint = len(segment_list) if (segment_list and len(segment_list) > 0) else 1
    frames_per_seg = max(input_latent.shape[0] // n_segs_hint, 1)  # default 64 when clip_size=64

    # ---- Derive segment texts from segment_list or fall back to Whisper ----
    if segment_list is not None and len(segment_list) > 0:
        seg_texts = [seg.get("text", "").strip() for seg in segment_list]
        n_segments = len(seg_texts)
        print(f"[Timeline] Processing {n_segments} speaker segments on timeline")
    else:
        result = whisper_model.transcribe(speaker_audio_path)
        print(result["text"])
        seg_texts = [result["text"].strip()]
        n_segments = 1

    # ---- Reshape input_latent into [n_segments, frames_per_seg, C, H, W] ----
    if input_latent.dim() == 4:
        total_frames_il = input_latent.shape[0]
        if total_frames_il == frames_per_seg * n_segments:
            seg_latents = input_latent.reshape(n_segments, frames_per_seg, *input_latent.shape[1:])
        elif total_frames_il == frames_per_seg:
            seg_latents = input_latent.unsqueeze(0)
        else:
            seg_latents = input_latent[:frames_per_seg].unsqueeze(0)
            n_segments = 1
            seg_texts = seg_texts[:1]
    else:
        seg_latents = input_latent[:1]
        n_segments = 1
        seg_texts = seg_texts[:1]

    # ---- Emotion features from first-segment latent ----
    _emo_device = next(encoder_emo.parameters()).device
    id_feats, emo_fea, au_disfa, au_bp4d = encoder_emo.forward_listener(seg_latents[0].to(_emo_device))
    expression = emo_fea['expression'].to(_emo_device)
    expression_soft = F.softmax(expression, dim=1)
    valence = emo_fea['valence'].to(_emo_device)
    arousal = emo_fea['arousal'].to(_emo_device)
    # ---- Listener reference transcription (once, for scoring) ----
    listener_result = whisper_model.transcribe(listener_audio_path)

    # ---- Rolling LLM dialogue: one turn per timeline segment ----
    random_number = 8
    tone_pos = 1
    all_text = ''
    current_text = ''
    _TTS_WORDS_PER_SEC = 1.5

    chat = [
        {"role": "system", "content": (
            "We are having a real-time conversation. Pretend you are a person with a specific personality "
            "(a sales representative), subject to the random seed: " + str(random_number) +
            ". For each speaker turn, generate a natural response that fits the live dialogue flow. "
            "Each reply must not exceed the word-count limit stated in the user message."
        )}
    ]

    seg_reply_texts = []
    for seg_i, speaker_text in enumerate(seg_texts):
        is_last_seg = (seg_i == len(seg_texts) - 1)
        if segment_list is not None and seg_i < len(segment_list):
            _seg_info = segment_list[seg_i]
            _seg_end = float(_seg_info.get('end', 0.64))
            if seg_i + 1 < len(segment_list):
                _next_start = float(segment_list[seg_i + 1].get('start', _seg_end))
                _gap_dur = max(0.5, _next_start - _seg_end)
            else:
                _gap_dur = None
        else:
            _gap_dur = frames_per_seg / 25.0

        if _gap_dur is None or is_last_seg:
            _max_words = None
        else:
            _max_words = max(5, round(_gap_dur * _TTS_WORDS_PER_SEC))
            _max_words = min(_max_words, 30)

        _word_limit_hint = f" [Reply in at most {_max_words} words.]" if _max_words is not None else ""
        if seg_i == 0:
            base_text = speaker_text if speaker_text else "Hello."
            user_msg = f"{base_text}{_word_limit_hint}"
        else:
            chat.append({"role": "assistant", "content": seg_reply_texts[-1]})
            emo_hint = (
                " (Please match the emotional tone — "
                "Neutral={:.2f}, Happy={:.2f}, Sad={:.2f}, valence={:.2f}, arousal={:.2f}. "
                "Do not include these scores in your reply.)"
            ).format(
                float(expression_soft[:, 0].mean()), float(expression_soft[:, 1].mean()),
                float(expression_soft[:, 2].mean()), float(valence.mean()), float(arousal.mean())
            )
            base_text = speaker_text if speaker_text else "Please continue."
            user_msg = f"{base_text}{emo_hint}{_word_limit_hint}"

        chat.append({"role": "user", "content": user_msg})
        response = pipe(chat, max_new_tokens=512)
        text_4_split = response[0]['generated_text'][-1]['content'].split('(')
        current_text = text_4_split[0]
        if len(text_4_split) >= 2:
            found = re.findall(r"\d+\.?\d*", text_4_split[1])
            if found:
                tone_pos = found[-1]
        print(f"[Seg {seg_i}] Speaker: {speaker_text!r}")
        print(f"[Seg {seg_i}] Listener: {current_text!r}")
        seg_reply_texts.append(current_text)
        all_text += current_text + " "

    chat.append({"role": "assistant", "content": seg_reply_texts[-1] if seg_reply_texts else ''})
    realism_score, sentiment_emb = evaluate_LLM(chat)

    bleu_scores = calculate_bleu_batch([listener_result["text"]], [all_text])
    rouge_scores = calculate_rouge_batch([listener_result["text"]], [all_text])
    bert_scores = calculate_bertscore_batch([listener_result["text"]], [all_text])
    evaluation_score = dict()
    evaluation_score['realism_score'] = realism_score
    evaluation_score['sentiment_emb'] = sentiment_emb
    evaluation_score['bleu_scores'] = bleu_scores[0]
    evaluation_score['bert_scores'] = bert_scores[0]

    # ---- TTS: timeline-aware audio (silence during speaker, TTS in gaps) ----
    OUTPUT_audio_FILE = result_img_save_path + '/output_audio' + str(batch_index) + '.wav'
    if not os.path.exists(result_img_save_path):
        os.makedirs(result_img_save_path)

    # Sample rate depends on TTS backend: ParlerTTS uses its model rate; EdgeTTS outputs 24 kHz
    if tts_method == 'edge':
        sr = 24000
    else:
        sr = audio_model.config.sampling_rate
    description_base = (
        "A female's speaker voice is monotone, with a very close recording that almost has no background noise. "
        "The tone is positive level of " + str(tone_pos)
    )
    _seg_list = segment_list if segment_list else [{'start': 0.0, 'end': 0.64, 'text': ''}]
    _sp_audio_info = sf.info(speaker_audio_path)
    total_duration_sec = _sp_audio_info.duration
    total_samples = int(sr * total_duration_sec)
    new_audio = np.zeros(total_samples, dtype=np.float32)

    # Pass 1: collect active segments that have a reply
    active_segs = []
    for seg_i, seg in enumerate(_seg_list):
        seg_end_sec = seg.get('end', 0.64)
        reply_text = seg_reply_texts[seg_i] if seg_i < len(seg_reply_texts) else ''
        next_start_sec = _seg_list[seg_i + 1].get('start', seg_end_sec) if seg_i + 1 < len(_seg_list) else total_duration_sec
        active_start_s = int(sr * seg_end_sec)
        active_end_s = min(total_samples, int(sr * next_start_sec))
        target_samples = active_end_s - active_start_s
        seg_dur = seg_end_sec - seg.get('start', 0.0)
        active_dur = max(0.0, next_start_sec - seg_end_sec)
        if not reply_text.strip() or target_samples <= 0:
            print(f'[Audio seg {seg_i}] silence={seg_dur:.2f}s  gap={active_dur:.2f}s  (no reply text)')
            continue
        active_segs.append({
            'seg_i': seg_i, 'reply_text': reply_text,
            'active_start_s': active_start_s, 'active_end_s': active_end_s,
            'target_samples': target_samples, 'seg_dur': seg_dur, 'active_dur': active_dur,
        })
    # pdb.set_trace()
    # Pass 2: generate TTS and place into new_audio
    if active_segs:
        if tts_method == 'edge':
            # EdgeTTS: generate each segment independently (faster, no GPU needed)
            print(f'[Audio] EdgeTTS generating {len(active_segs)} segments.')
            for seg_info in active_segs:
                _tmp_mp3 = os.path.join(result_img_save_path, f'{batch_index}_edge_{seg_info["seg_i"]}.mp3')
                _tmp_wav = os.path.join(result_img_save_path, f'{batch_index}_edge_{seg_info["seg_i"]}.wav')
                communicate = edge_tts.Communicate(seg_info['reply_text'], VOICE)
                communicate.save_sync(_tmp_mp3)
                import subprocess as _sp
                _sp.run(['ffmpeg', '-y', '-v', 'warning', '-i', _tmp_mp3,
                         '-ar', str(sr), '-ac', '1', _tmp_wav], check=True)
                os.remove(_tmp_mp3)
                _seg_audio, _ = sf.read(_tmp_wav, dtype='float32')
                if _seg_audio.ndim > 1:
                    _seg_audio = _seg_audio[:, 0]
                os.remove(_tmp_wav)
                place_len = min(len(_seg_audio), seg_info['active_end_s'] - seg_info['active_start_s'])
                new_audio[seg_info['active_start_s']:seg_info['active_start_s'] + place_len] = _seg_audio[:place_len]
                print(f'[Audio seg {seg_info["seg_i"]}] EdgeTTS silence={seg_info["seg_dur"]:.2f}s  '
                      f'gap={seg_info["active_dur"]:.2f}s  reply={repr(seg_info["reply_text"][:60])}')
        else:
            # ParlerTTS: generate all segments in one call then split by char proportion
            concat_text = "      ".join(s['reply_text'] for s in active_segs)
            print(f'[Audio] Generating TTS for {len(active_segs)} segments. text={repr(concat_text[:120])}')
            _input_ids = tts_tokenizer(description_base, return_tensors="pt").input_ids.to(audio_device)
            _prompt_ids = tts_tokenizer(concat_text, return_tensors="pt").input_ids.to(audio_device)
            tts_all = audio_model.generate(input_ids=_input_ids, prompt_input_ids=_prompt_ids).cpu().numpy().squeeze().astype(np.float32)
            total_tts_samples = len(tts_all)

            char_lengths = [len(s['reply_text']) for s in active_segs]
            total_chars = max(sum(char_lengths), 1)
            cumulative = 0
            tts_ptr = 0
            for idx, seg_info in enumerate(active_segs):
                cumulative += char_lengths[idx] + 28
                tts_end = int(total_tts_samples * cumulative / total_chars) if idx < len(active_segs) - 1 else total_tts_samples
                tts_seg = tts_all[tts_ptr:tts_end].copy()
                tts_ptr = tts_end
                place_len = min(len(tts_seg), seg_info['active_end_s'] - seg_info['active_start_s'])
                new_audio[seg_info['active_start_s']:seg_info['active_start_s'] + place_len] = tts_seg[:place_len]
                print(f'[Audio seg {seg_info["seg_i"]}] silence={seg_info["seg_dur"]:.2f}s  '
                      f'gap={seg_info["active_dur"]:.2f}s  reply={repr(seg_info["reply_text"][:60])}')
    # pdb.set_trace()
    sf.write(OUTPUT_audio_FILE, new_audio, sr)
    print('audio_generated, path is: ', OUTPUT_audio_FILE)

    # ---- UTMOS MOS evaluation (kept from evaluate_final.py) ----
    if utmos_available:
        try:
            audio_data_utmos, _ = librosa.load(OUTPUT_audio_FILE, sr=16000)
            mos_score = utmos_model.predict(data=audio_data_utmos, sr=16000)
            print(f"UTMOS MOS Score: {mos_score}")
            evaluation_score['mos_score'] = float(mos_score) if hasattr(mos_score, 'item') else float(mos_score)
        except Exception as e:
            print(f"UTMOS evaluation failed: {e}")
            evaluation_score['mos_score'] = None
    else:
        evaluation_score['mos_score'] = None
    # ---- Whisper feature extraction for talking head ----
    whisper_feature = audio_processor.audio2feat(OUTPUT_audio_FILE)
    whisper_chunks = audio_processor.feature2chunks(feature_array=whisper_feature, fps=fps)
    # pdb.set_trace()
    # ---- Load listener frames: real video or fall back to generated latent ----
    if listener_video_path is not None:
        _n_frames = listener_num_frames if listener_num_frames is not None else input_latent.shape[0]
        print(f'[input_latent_talk] loading {_n_frames} frames from real listener video: {listener_video_path}')
        _lv_clip, _lv_total = load_frames_from_video_path(listener_video_path, 50, _n_frames, stride=1)
        input_latent_talk = _lv_clip.squeeze(0).to(input_latent.device)
        print(f'[input_latent_talk] loaded {input_latent_talk.shape[0]}/{_lv_total} frames')
    else:
        input_latent_talk = input_latent

    input_img_list = np.split(
        np.uint8(np.transpose(127.5 * (input_latent_talk + 1).cpu().numpy(), (0, 2, 3, 1)))[:, :, :, ::-1],
        input_latent_talk.shape[0], axis=0
    )
    bbox_shift = 0
    coord_list_single, frame_list_single = get_landmark_and_bbox(input_img_list, bbox_shift)
    # pdb.set_trace()
    # ---- Reverse-and-repeat x3 for temporal diversity ----
    repeat_times = 2 #remember to set back to 3 for final evaluation
    coord_list = repeat_times * (coord_list_single + coord_list_single[::-1])
    frame_list  = repeat_times * (frame_list_single  + frame_list_single[::-1])
    print('len(frame_list)', len(frame_list))

    input_latent_list = []
    croped_image = []
    for bbox, frame in zip(coord_list, frame_list):
        bbox = np.array(bbox)
        bbox[bbox < 0] = 0
        frame = np.squeeze(frame)
        if (bbox == coord_placeholder).all():
            x1, y1, x2, y2 = [0, 0, 256, 256]
        else:
            x1, y1, x2, y2 = bbox
        crop_frame = frame[y1:y2, x1:x2]
        crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
        latents = vae.get_latents_for_unet(crop_frame)
        croped_image.append(torch.unsqueeze(torch.tensor(crop_frame), dim=0))
        input_latent_list.append(latents)
    # pdb.set_trace()
    input_latent_list_cycle = input_latent_list

    new_frame_list = []
    for single_ori_img in frame_list:
        new_frame_list.append(
            torch.permute(torch.tensor(single_ori_img[:, :, :, ::-1] / 127.5 - 1), (0, 3, 1, 2))
        )
    original_image_cycle = new_frame_list

    laten_repeat = max(len(whisper_chunks) // len(input_latent_list_cycle), 1)
    vae_encode_latents_new = []
    for single_latent in original_image_cycle:
        vae_encode_latents_new += laten_repeat * [single_latent]
    original_image_cycle = vae_encode_latents_new

    coord_list_cycle = coord_list
    coord_list_cycle_new = []
    for single_coord in coord_list_cycle:
        coord_list_cycle_new += laten_repeat * [single_coord]
    coord_list_cycle = coord_list_cycle_new

    # ---- Talking head forward pass ----
    video_num = len(whisper_chunks)
    timesteps = torch.tensor([0], device=unet.device)
    gen = datagen_reaction(whisper_chunks, input_latent_list_cycle, batch_size)

    # pdb.set_trace()
    res_frame_list = []
    res_frame_list_latent = []
    res_frame_list_tensor = []
    for i, (whisper_batch, latent_batch) in enumerate(tqdm(gen, total=int(np.ceil(float(video_num) / batch_size)))):
        audio_feature_batch = torch.from_numpy(whisper_batch)
        audio_feature_batch = audio_feature_batch.to(device=unet.device, dtype=unet.model.dtype)
        audio_feature_batch = pe(audio_feature_batch)
        latent_batch = latent_batch.to(dtype=unet.model.dtype)
        pred_latents = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature_batch).sample
        recon = vae.decode_latents(pred_latents)
        for res_frame in recon:
            res_frame_list.append(res_frame)
            res_frame_list_tensor.append(torch.unsqueeze(torch.tensor(res_frame.copy()), dim=0))
        for res_latent in pred_latents:
            res_frame_list_latent.append(res_latent)

    # Save intermediate cropped talking-head video
    # croped_image_tensor = torch.cat(res_frame_list_tensor, dim=0).to(dtype=torch.uint8).cpu().contiguous()
    # video_save_path = os.path.join(args.save_video_path, 'sample_test_afterRF_crop' + str(batch_index) + '.mp4')
    # print(croped_image_tensor.shape)
    # imageio.mimwrite(video_save_path, croped_image_tensor, fps=8, quality=9)
    # print('save path {}'.format(args.save_video_path))
    # pdb.set_trace()
    if return_image4emotion:
        interval_listener = len(res_frame_list) // num_frame_sp
        out_images = res_frame_list if interval_listener == 0 else res_frame_list[::interval_listener]
        return out_images[:num_frame_sp], evaluation_score, OUTPUT_audio_FILE

    # ---- Blend talking-head crop back into original frames ----
    print("pad talking image to original video")
    final_output_frames = []
    for i, res_frame in enumerate(tqdm(res_frame_list)):
        ori_frame = torch.squeeze(copy.deepcopy(original_image_cycle[i % len(original_image_cycle)]))
        bbox = np.array(coord_list_cycle[i % len(coord_list_cycle)])
        bbox[bbox < 0] = 0
        if (bbox == coord_placeholder).all():
            bbox = [0, 0, 256, 256]
        x1, y1, x2, y2 = bbox
        res_frame = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
        combine_frame = get_image(
            np.uint8(127.5 * (ori_frame.permute(1, 2, 0).cpu() + 1))[:, :, ::-1], res_frame, bbox
        )
        cv2.imwrite(f"{result_img_save_path}/{str(i).zfill(8)}.png", combine_frame)
        final_output_frames.append(combine_frame[:, :, ::-1])

    cmd_img2video = (f"ffmpeg -y -v warning -r {fps} -f image2 -i {result_img_save_path}/%08d.png"
                     f"  -vf format=rgb24,scale=out_color_matrix=bt709,format=yuv420p -crf 18 temp2.mp4")
    print(cmd_img2video)
    os.system(cmd_img2video)

    output_vid_name = os.path.join(args.save_video_path, 'sample_test_final' + str(batch_index) + '.mp4')
    cmd_combine_audio = f"ffmpeg -y -v warning -i {OUTPUT_audio_FILE} -i temp2.mp4 {output_vid_name}"
    print(cmd_combine_audio)
    os.system(cmd_combine_audio)

    for _png in sorted(os.listdir(result_img_save_path)):
        if _png.endswith('.png'):
            os.remove(os.path.join(result_img_save_path, _png))
    print(f"result is save to {output_vid_name}")
    print('the corresponding speaker path is: ', speaker_audio_path)

    out_images = final_output_frames
    return out_images[:num_frame_sp], evaluation_score, OUTPUT_audio_FILE

def calculate_talkinghead_genall(audio_file,whisper_model,audio_processor,emo_label,input_latent, pipe,vae, unet, pe,batch_index=0,batch_size=8,fps=25,
        result_img_save_path='./temp',VOICE = "en-GB-SoniaNeural"):
    # model = whisper.load_model("base.en")

    # pdb.set_trace()
    #get emotion from images
    _emo_device = next(encoder_emo.parameters()).device
    id_feats,emo_fea,au_disfa,au_bp4d=encoder_emo.forward_listener(input_latent.to(_emo_device))
    expression=emo_fea['expression']
    expression_soft=F.softmax(expression,dim=1)
    valence=emo_fea['valence']
    arousal=emo_fea['arousal']


    # pdb.set_trace()
    result = whisper_model.transcribe(audio_file)
    print(result["text"])
    
    sentences=result["text"].split('.')

    rand_num=torch.randint(0,100,(1,))
    all_text=''
    conversation_history = f"system: You are a person with a specific personality, subjected to the random seed:."+str(5)+". reply within 30 words\n"
    # for i in range(1):
    #     if i==0:
    #         chat = [
    #         {"role": "system", "content": "You are a person with a specific personality, subjected to the random seed:."+str(5)+". reply within 30 words"},
    #         {"role": "user", "content": sentences[0]}
    #             ]
    #         a=1
    #     else:
    #         chat=response[0]['generated_text']
    #         chat.append(
    #             # {"role": "system", "content": "You are a person with a specific personality, subjected to the random number:."+str(6)+". reply with 10 words"},
    #             {"role": "user", "content": sentences[i]+". please reply within 10 words"}
    #         )
    #     response = pipe(chat, max_new_tokens=512)#,pad_token_id=pipe.tokenizer.eos_token_id
    #     current_text=response[0]['generated_text'][-1]['content']
    #     all_text=all_text+current_text
    #     print(response[0]['generated_text'][-1]['content'])
    # pdb.set_trace()
    for i in range(len(sentences)):
        if i==0:
            chat = [
            {"role": "system", "content": "We are having an interview, pretend you are a person with a specific personality, subject to the random seed:"+str(5)\
            +".Please stay tune for more information if the user input is not clear. Please generate an appropriate response within 10 words( more than 5 words) in each reply"},
            
            # {"role": "system", "content": 
            # "Please reply with a tone of emotion level:\
            # Neutral:{single_express[0]}, Happy:{single_express[1]}, 'Sad':{single_express[2]}, 'Surprise':{single_express[3]},\
            #                     'Fear':{single_express[4]}, 'Disgust':{single_express[5]}, 'Anger':{single_express[6]}, 'Contempt':{single_express[7]}, \
            #             'valence':{single_val}, 'arousal':{single_arousal}. For example Happy=0 means not happy at all, Happy=1 means extremely happy"},            
            
            
            
            {"role": "user", "content": sentences[0]+". (Please reply with a tone of the following emotion degree:\
            Neutral:{}, Happy:{}, 'Sad':{}, 'Surprise':{}, 'Fear':{}, 'Disgust':{}, 'Anger':{}, 'Contempt':{}, 'valence':{}, 'arousal':{}. For example Happy=0 means not happy at all, Happy=1 means extremely happy".format(
                            expression_soft[:,0].mean(),expression_soft[:,1].mean(),expression_soft[:,2].mean(),expression_soft[:,3].mean(),expression_soft[:,4].mean(),
                            expression_soft[:,5].mean(),expression_soft[:,6].mean(),expression_soft[:,7].mean(),valence.mean(),arousal.mean())
                        +'. Remove the emotion number in your reply.)'}
                ]
            a=1
        else:
            # chat=response[0]['generated_text']
            chat.append(
            {"role": "Assistant", "content": current_text})
            chat.append(
                # {"role": "system", "content": "You are a person with a specific personality, subjected to the random number:."+str(6)+". reply with 10 words"},
                {"role": "user", "content": sentences[i]+". (Please reply with a tone of the following emotion degree:\
            Neutral:{}, Happy:{}, 'Sad':{}, 'Surprise':{}, 'Fear':{}, 'Disgust':{}, 'Anger':{}, 'Contempt':{}, 'valence':{}, 'arousal':{}. For example Happy=0 means not happy at all, Happy=1 means extremely happy".format(
                            expression_soft[:,0].mean(),expression_soft[:,1].mean(),expression_soft[:,2].mean(),expression_soft[:,3].mean(),expression_soft[:,4].mean(),
                            expression_soft[:,5].mean(),expression_soft[:,6].mean(),expression_soft[:,7].mean(),valence.mean(),arousal.mean())
                        +'. Remove the emotion number in your reply.)'}#+". please reply within 10 words"}
            )
        
        response = pipe(chat, max_new_tokens=512)#,pad_token_id=pipe.tokenizer.eos_token_id
        # pdb.set_trace()
        current_text=response[0]['generated_text'][-1]['content']
        all_text=all_text+current_text
        print(response[0]['generated_text'][-1]['content'])
    
    # # #text to audio
    # # pdb.set_trace()
    # for i in range(len(sentences)):
    #     if i==0:
    #         chat = [
    #         {"role": "system", "content": "We are having an interview. You are a person with a specific personality, subjected to the random seed:."+str(5)+". Please reply within 10 words"},
    #         {"role": "user", "content": sentences[0]}
    #             ]
    #         a=1
    #         conversation_history += f"User: {sentences[0]}\nAssistant:"
    #     else:
    #         # chat=response[0]['generated_text']
    #         # chat.append(
    #         #     # {"role": "system", "content": "You are a person with a specific personality, subjected to the random number:."+str(6)+". reply with 10 words"},
    #         #     {"role": "user", "content": sentences[i]+". please reply within 10 words"}
    #         # )
    #         conversation_history += f"{response}\n"
    #         conversation_history += f"User: {sentences[i]}\nAssistant:"
    #     # conversation_history += f"User: {user_input}\n"
    #     pdb.set_trace()
    #     # Generate a response using the pipeline
    #     response = pipe(conversation_history, max_length=1000)[0]["generated_text"]
        
    #     # Extract the new response part (after the user input)
    #     response = response[len(conversation_history):]
    #     # all_text=all_text+response
    #     print(f"Assistant: {response.strip()}")

    #     # # Append LLaMA's response to the conversation history
    #     # conversation_history += f"LLaMA: {response}\n"
    
    # from transformers import AutoModelForCausalLM, AutoTokenizer

    # # Load the Llama 3 model and tokenizer (assuming it's hosted on Hugging Face)
    # model_name = 'meta-llama/Llama-3'
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModelForCausalLM.from_pretrained(model_name)
    # from transformers import LlamaForCausalLM, LlamaTokenizer
    # tokenizer = LlamaTokenizer.from_pretrained("/path/to/local/llama-3")
    # model = LlamaForCausalLM.from_pretrained("/path/to/local/llama-3")
    

    # conversation_history = ""

    # while True:
    #     # Get user input
    #     user_input = input("You: ")
        
    #     # Append user input to conversation history
    #     conversation_history += f"User: {user_input}\n"
        
    #     # Get LLaMA's response
    #     response = chat_with_llama(model, tokenizer, conversation_history)
    #     print(f"LLaMA: {response}")
        
    #     # Append the model's response to conversation history
    #     conversation_history += f"LLaMA: {response}\n"
    # # wavs = chat.infer(current_text)


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

    # # #image processing
    # # # crop_frame = cv2.imread(img_path)
    # # #detect landmark
    # img_path='/homes/zs003/test_images/siyang.png'
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
    # crop_frame=cv2.resize(crop_frame,(256,256),interpolation = cv2.INTER_LANCZOS4)
    # latents = vae.get_latents_for_unet(crop_frame)
    # input_latent_list=[]
    # input_latent_list.append(latents)

    
    # input_img_list = torch.chunk(input_latent.cpu(),input_latent.shape[0],dim=0)#[img_path, ]
    input_img_list = np.split(np.uint8(np.transpose(127.5*(input_latent+1).cpu().numpy(),(0,2,3,1)))[:,:,:,::-1],input_latent.shape[0],axis=0)#[img_path, ]
    # input_img_list[sample]
    bbox_shift=0
    coord_list, frame_list = get_landmark_and_bbox(input_img_list, bbox_shift)
    # with open(crop_coord_save_path, 'wb') as f:
    #     pickle.dump(coord_list, f)
    input_latent_list = []
    # pdb.set_trace()
    for bbox, frame in zip(coord_list, frame_list):
        frame=np.squeeze(frame)
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
    # pdb.set_trace()
    # crop_frame=cv2.resize(crop_frame,(256,256),interpolation = cv2.INTER_LANCZOS4)
    # latents = vae.get_latents_for_unet(crop_frame)
    # input_latent_list=[]
    # input_latent_list.append(latents)

    # #no crop
    # pdb.set_trace()
    # input_latent_list=[]
    # for single_frame in input_latent:
    #     # crop_frame=cv2.resize(single_frame,(256,256),interpolation = cv2.INTER_LANCZOS4)
    #     latents = vae.get_latents_for_unet_nonorm(single_frame)
    #     input_latent_list.append(latents)

    #input latent start, directly use latent
    # input_latent_list=input_latent
    # input_latent_list_cycle=torch.cat((input_latent_list,torch.flip(input_latent_list,[0])),dim=0)

    #----
    #use image----
    input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
    original_image_cycle= np.split(input_latent,input_latent.shape[0],axis=0)+ np.split(input_latent,input_latent.shape[0],axis=0)[::-1]
    
    laten_repeat=len(whisper_chunks)//len(input_latent_list_cycle)
    # pdb.set_trace()
    vae_encode_latents_new=[]
    for single_latent in original_image_cycle:
        repeated_latent=laten_repeat * [single_latent]
        vae_encode_latents_new+=repeated_latent
    original_image_cycle=vae_encode_latents_new
    
    # frame_list_cycle = frame_list + frame_list[::-1]
    coord_list_cycle = coord_list + coord_list[::-1]

    coord_list_cycle_new=[]
    for single_coord in coord_list_cycle:
        repeated_coord=laten_repeat * [single_coord]
        coord_list_cycle_new+=repeated_coord
    coord_list_cycle=coord_list_cycle_new


    video_num = len(whisper_chunks)
    # input_latent_list_cycle = [val for val in input_latent_list for i in range(len(whisper_chunks))]
    # input_latent_list_cycle= len(whisper_chunks)*input_latent_list 
    # pdb.set_trace()
    timesteps = torch.tensor([0], device=unet.device)
    # gen = datagen(whisper_chunks,input_latent_list_cycle,batch_size)
    gen = datagen_reaction(whisper_chunks,input_latent_list_cycle,batch_size)
    
    res_frame_list=[]
    res_frame_list_latent=[]
    # pdb.set_trace()
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
        recon = vae.decode_latents(pred_latents)
        
        # pdb.set_trace()
        for res_frame in recon:
            res_frame_list.append(res_frame)
        for res_latent in pred_latents:
            #mask latent
            # res_latent_detrans=res_latent*0.5+0.5
            # res_latent_detrans = res_latent_detrans * (self._mask_tensor>0.5)
            res_frame_list_latent.append(res_latent)
            # cv2.imwrite(f"{result_img_save_path}/{str(i).zfill(8)}.png",res_frame)
    # return res_frame_list




    print("pad talking image to original video")
    for i, res_frame in enumerate(tqdm(res_frame_list)):
        
        ori_frame = torch.squeeze(copy.deepcopy(original_image_cycle[i%(len(original_image_cycle))]))#.permute(2,1,0)
        # ori_frame = copy.deepcopy(res_frame_list[i%(len(res_frame_list))])
        # t_shape = ori_frame.shape#coord_list_cycle[i%(len(coord_list_cycle))]
        t_shape = coord_list_cycle[i%(len(coord_list_cycle))]
        bbox=[0,0,t_shape[0],t_shape[1]]
        x1, y1, x2, y2 = bbox
        # try:
        res_frame = cv2.resize(res_frame.astype(np.uint8),(x2-x1,y2-y1))
        # pdb.set_trace()
        # combine_frame = get_image(ori_frame,res_frame,bbox)
        # combine_frame = get_image(np.uint8(127.5*(ori_frame.permute(1,2,0).cpu()+1))[:,:,::-1],res_frame,bbox)
        combine_frame = get_image(np.uint8(127.5*(ori_frame.permute(1,2,0).cpu()+1))[:,:,::-1],res_frame,bbox)

        cv2.imwrite(f"{result_img_save_path}/{str(i).zfill(8)}.png",combine_frame)
        # pdb.set_trace()
        # cv2.imwrite(f"{result_img_save_path}/{str(i).zfill(8)}.png",ori_frame)
        # cv2.imwrite(f"{result_img_save_path}/{str(i).zfill(8)}.png",res_frame)
    cmd_img2video = f"ffmpeg -y -v warning -r {fps} -f image2 -i {result_img_save_path}/%08d.png  -vf format=rgb24,scale=out_color_matrix=bt709,format=yuv420p -crf 18 temp2.mp4"

    print(cmd_img2video)
    os.system(cmd_img2video)
    

    # output_vid_name = os.path.join('./result_talking/siyang',"siyang.mp4")
    output_vid_name = os.path.join(args.save_video_path, 'sample_test_final' + str(batch_index)+'.mp4')
    
    cmd_combine_audio = f"ffmpeg -y -v warning -i {OUTPUT_audio_FILE} -i temp2.mp4 {output_vid_name}"
    print(cmd_combine_audio)
    os.system(cmd_combine_audio)
    
    # os.remove("./temp.mp4")
    shutil.rmtree(result_img_save_path)
    print(f"result is save to {output_vid_name}")      

    # print('the corresponding speaker path is: ', speaker_audio_path)
    # pdb.set_trace()
    # if return_image4emotion:
    #return emotion image
    # pdb.set_trace()
    interval_listener=len(res_frame_list)//num_frame_sp
    out_images=res_frame_list[::interval_listener]
    # return out_images[:num_frame_sp],evaluation_score 
    
    return out_images[:num_frame_sp],evaluation_score 
def llm_select_response_segments(pipe, full_transcript, segments, max_segments=1):#remember to change back to 3
    """Use LLM to decide which speaker segments warrant a listener verbal response.
    Returns a filtered (subset) list of segment dicts from *segments*.
    """
    if not segments:
        return segments
    seg_desc = ""
    for i, seg in enumerate(segments):
        seg_desc += (f"  Segment {i}: [{seg.get('start',0):.2f}s – "
                     f"{seg.get('end',0):.2f}s] \"{seg.get('text','')}\"\n")
    prompt_text = (
        "You are a conversation analyst. Given the following speaker transcript "
        "broken into timed segments, decide at which segments the listener should "
        "produce a verbal response (e.g. acknowledgment, question, comment). "
        "Consider natural conversational turn-taking cues such as questions, "
        "pauses, topic shifts, and emotional moments.\n\n"
        f"Full transcript: \"{full_transcript}\"\n\n"
        f"Segments:\n{seg_desc}\n"
        "Return ONLY a JSON list of segment indices (0-based) where the listener "
        "should respond.  Example: [0, 2]\n"
        "Output:"
    )
    chat_msgs = [
        {"role": "system", "content": "You are a helpful assistant that outputs only valid JSON."},
        {"role": "user", "content": prompt_text}
    ]
    response = pipe(chat_msgs, max_new_tokens=128)
    raw = response[0]['generated_text'][-1]['content'].strip()
    try:
        indices = _json.loads(raw)
        if not isinstance(indices, list):
            indices = list(range(len(segments)))
    except (_json.JSONDecodeError, ValueError):
        nums = re.findall(r'\d+', raw)
        indices = [int(n) for n in nums if int(n) < len(segments)]
        if not indices:
            indices = list(range(len(segments)))

    # Deduplicate and clamp
    seen = set()
    unique_indices = []
    for idx in indices:
        idx = int(idx)
        if 0 <= idx < len(segments) and idx not in seen:
            seen.add(idx)
            unique_indices.append(idx)
    indices = unique_indices
    print(f"[LLM Timing] Raw output: {raw!r}  →  indices: {indices}")

    # Prefer segments followed by the largest silence gap (most natural response window)
    if len(indices) > 1:
        gap_info = []
        for idx in indices:
            seg_end = segments[idx].get('end', 0.0)
            next_start = segments[idx + 1].get('start', seg_end) if idx + 1 < len(segments) else seg_end + 999.0
            gap_info.append((max(0.0, next_start - seg_end), idx))
        max_gap = max(g for g, _ in gap_info)
        gap_threshold = max_gap * 0.8
        indices = [idx for gap, idx in gap_info if gap >= gap_threshold]
        print(f"[LLM Timing] Gap filter → kept indices: {indices}")

    # Cap to max_segments: keep the ones with the largest following gaps
    if max_segments is not None and len(indices) > max_segments:
        gap_info = []
        for idx in indices:
            seg_end = segments[idx].get('end', 0.0)
            next_start = segments[idx + 1].get('start', seg_end) if idx + 1 < len(segments) else seg_end + 999.0
            gap_info.append((max(0.0, next_start - seg_end), idx))
        gap_info.sort(key=lambda x: -x[0])
        kept = sorted(idx for _, idx in gap_info[:max_segments])
        print(f"[LLM Timing] Max-segment cap ({max_segments}) → kept indices: {kept}")
        indices = kept

    selected = [segments[i] for i in indices if 0 <= i < len(segments)]
    if not selected:
        selected = segments
    return selected


def main(args):
    # # Setup PyTorch:
    # # torch.manual_seed(args.seed)
    # setup_distributed()
    # # dist.init_process_group("nccl")
    # # assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    # # rank = dist.get_rank()
    # # device = rank % torch.cuda.device_count()
    # # local_rank = rank

    # rank = int(os.environ["RANK"])
    # local_rank = int(os.environ["LOCAL_RANK"])
    # device = torch.device("cuda", local_rank)
    device = torch.device("cuda", 0)
    # device = torch.device("cuda", 0)
    clip_size=args.num_frames #64 32
    is_2025 = True  # set to False to use react2024 metrics
    audio_file='/import/nobackup_mmv_ioannisp/zs003/face_react/new_audio_folder/all_together/NoXI/026_2016-04-06_Nottingham/Expert_video/3.wav'
    listener_audio_file='/import/nobackup_mmv_ioannisp/zs003/face_react/new_audio_folder/all_together/NoXI/026_2016-04-06_Nottingham/Novice_video/3.wav'
    #
    # pdb.set_trace()
    print('check test/val matrix..')
    # if val_test == 'val':
    if 'sbg'  in platform.node() or 'dg' in platform.node() or 'xlg' in platform.node():
        args.dataset_path='/data/home/acw569/paper_face/face_react/latte/latte_reaction/val_test_matrix'
    else:
        # args.dataset_path='/import/nobackup_mmv_ioannisp/zs003/code/face_reaction/latte_reaction/val_test_matrix'
        args.dataset_path='./val_test_matrix'
    # speaker_neighbour_matrix = np.load(os.path.join(args.dataset_path, 'neighbour_emotion_val.npy'))
    # # else:
    # speaker_neighbour_matrix = np.load(os.path.join(args.dataset_path, 'neighbour_emotion_test.npy'))

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

    _t_load_start = time.time()
    #load audio->talking head branch--------------------------------
    whisper_model = whisper.load_model("base.en")

    # result = whisper_model.transcribe(audio_file)
    # print(result["text"])

    # sentences=result["text"].split('.')
    ##LLM processing
    llm_device = 1 if torch.cuda.device_count() > 1 else 0
    if 'server' in platform.node():
        pipe = pipeline("text-generation", model="/data/szl/tools/llama3", torch_dtype=torch.float16, device=llm_device)
    else:
        pipe = pipeline("text-generation", "meta-llama/Meta-Llama-3-8B-Instruct", torch_dtype=torch.float16, device=llm_device)

    ### deepseek
    # pipe = pipeline("text-generation", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", torch_dtype=torch.float16, device=1)
    
    #initlize chat
    
    #init talking head
    
    # load model weights
    # pdb.set_trace()
    audio_processor, vae, unet, pe = load_all_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timesteps = torch.tensor([0], device=device)


    print(f"[Time] Whisper + LLM pipeline loading: {time.time()-_t_load_start:.2f}s")
    #end audio->talking head----------------------------------------
    using_cfg = args.cfg_scale > 1.0

    # Load model:
    _t_model_start = time.time()
    latent_size = args.image_size // 8
    args.latent_size = latent_size
    model = get_models(args).to(device)

    if args.use_compile:
        model = torch.compile(model)

    # a pre-trained model or load a custom Latte checkpoint from train.py:
    ckpt_path = args.ckpt
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict,strict=True)

    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    # vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    reaction_vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae").to(device)


    # vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae").to(device)
    # text_encoder = TextEmbedder().to(device)

    if args.use_fp16:
        print('WARNING: using half percision for inferencing!')
        vae.to(dtype=torch.float16)
        model.to(dtype=torch.float16)
        # text_encoder.to(dtype=torch.float16)
    print(f"[Time] Latte model + VAE loading: {time.time()-_t_model_start:.2f}s")
    dataset = get_dataset(args,train_mode='test')


    
    # sampler = DistributedSampler(
    # dataset,
    # num_replicas=dist.get_world_size(),
    # rank=0,
    # shuffle=True,
    # seed=args.global_seed
    # )
    batch_size=int(args.local_batch_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=None,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    listener_emotion_pred_list=[]
    listener_emotion_gt_list=[]
    speaker_emotion_gt_list=[]
    all_listener_emotion_pred_list=[]
    all_score=dict()
    # all_score['']

    all_score['all_realism_score']=0
    all_score['all_sentiment_score']=0
    all_score['all_valence_score']=0
    all_score['all_arousal_score']=0
    all_score['all_bleu_scores']=0
    all_score['all_rouge_scores']=0
    all_score['all_bert_scores']=0
    all_score['all_mos_scores']=0

    all_score['all_sentiment_emb']=torch.zeros([len(dataset),10])
    # pdb.set_trace()
    # Labels to condition the model with (feel free to change):
    double_count=0
    for step, video_data in enumerate(loader):
        if step<300:
            print('step jump', step)
            continue
        _t_step_start = time.time()
        print('current step is:{}, batch_size is :{}'.format(step,batch_size))

        # ── Resume: if this step's checkpoint exists, reload state and skip ──
        _step_ckpt = os.path.join(args.save_video_path, f'ckpt_step_{step:06d}.pt')
        if os.path.exists(_step_ckpt):
            _c = torch.load(_step_ckpt, map_location='cpu')
            listener_emotion_pred_list.append(_c['listener_emotion_pred'])
            listener_emotion_gt_list.append(_c['listener_emotion_gt'])
            speaker_emotion_gt_list.append(_c['speaker_emotion_gt'])
            print(f"[Resume] Step {step} already done — loaded {_step_ckpt}, skipping.")
            continue
        torch.manual_seed(0)
        out_samples=None
        # pdb.set_trace()
        if args.use_fp16:
            z = torch.randn(batch_size, args.num_frames, 4, latent_size, latent_size, dtype=torch.float16, device=device) # b c f h w
        else:
            z = torch.randn(batch_size, args.num_frames, 4, latent_size, latent_size, device=device)
        # if args.use_fp16:
        #     z = torch.randn(batch_size, args.num_frames, 4, latent_size, latent_size, dtype=torch.float16, device=device) # b c f h w
        # else:
        #     z = torch.randn(batch_size, args.num_frames, 4, latent_size, latent_size, device=device)

        # Setup classifier-free guidance:
        # z = torch.cat([z, z], 0)
        if using_cfg:
            # pdb.set_trace()
            ###original classifier free guidance
            # z = torch.cat([z, z], 0)
            # y = torch.randint(0, args.num_classes, (1,), device=device)
            # y_null = torch.tensor([101] * 1, device=device)
            # y = torch.cat([y, y_null], dim=0)
            # model_kwargs = dict(y=y, cfg_scale=args.cfg_scale, use_fp16=args.use_fp16)
            # sample_fn = model.forward_with_cfg
            #-----------------end original guidance
            
            #-----------------begin reaction guidance
            z = torch.cat([z, z], 0)
            # y = torch.randint(0, args.num_classes, (1,), device=device)
            # y_null = torch.tensor([101] * 1, device=device)
            # y = torch.cat([y, y_null], dim=0)
            model_kwargs = dict(y=None, cfg_scale=args.cfg_scale, use_fp16=args.use_fp16)
            sample_fn = model.forward_with_cfg_reaction
        else:
            sample_fn = model.forward
            model_kwargs = dict(y=None, use_fp16=args.use_fp16)


        # ── Whisper-based segment selection (adapted from demo_ct_v4.py) ──────
        # Transcribe speaker audio to get timed segments, then use the LLM to
        # select which segments need a listener verbal response.
        _sp_audio_path_0 = video_data['speaker_audio_path'][0]
        _t_whisper = time.time()
        _whisper_result = whisper_model.transcribe(_sp_audio_path_0)
        _all_segments = _whisper_result.get("segments", [])
        print(f"[Time] Step {step} | Whisper transcription: {time.time()-_t_whisper:.2f}s")
        if len(_all_segments) == 0:
            _all_segments = [{"start": 0.0, "end": clip_size / 25.0, "text": ""}]

        _full_transcript = " ".join([s.get("text", "") for s in _all_segments])
        _t_llm_seg = time.time()
        _selected_segs = llm_select_response_segments(pipe, _full_transcript, _all_segments)
        print(f"[Time] Step {step} | LLM segment selection: {time.time()-_t_llm_seg:.2f}s")

        # Build cumulative text per selected segment so each LLM reply covers
        # ALL speaker content since the previous response (mirrors demo_ct_v4.py).
        _selected_with_cumulative = []
        _prev_orig_idx = -1
        _last_orig_idx = -1
        for _sel_seg in _selected_segs:
            _cur_orig_idx = next((i for i, s in enumerate(_all_segments) if s is _sel_seg), None)
            if _cur_orig_idx is None:
                _selected_with_cumulative.append(_sel_seg)
                continue
            _span_text = " ".join(
                s.get("text", "").strip()
                for s in _all_segments[_prev_orig_idx + 1 : _cur_orig_idx + 1]
            )
            _new_seg = dict(_sel_seg)
            _new_seg["text"] = _span_text
            _selected_with_cumulative.append(_new_seg)
            _prev_orig_idx = _cur_orig_idx
            _last_orig_idx = _cur_orig_idx
        if _last_orig_idx + 1 < len(_all_segments):
            _tail_text = " ".join(
                s.get("text", "").strip() for s in _all_segments[_last_orig_idx + 1 :]
            )
            _tail_seg = dict(_all_segments[-1])
            _tail_seg["text"] = _tail_text
            _selected_with_cumulative.append(_tail_seg)
        segments = _selected_with_cumulative
        print(f"[Step {step}] Processing {len(segments)} listener-response segments")

        # ── Per-segment diffusion generation ────────────────────────────────
        # For each selected segment we run the diffusion model on the
        # corresponding video slice, then concatenate all clips.
        _t_diffusion = time.time()
        seg_samples_list = []
        emo_label_list = []
        listener_ref_list = []
        speaker_emo_list = []

        total_data_frames = video_data['img'].shape[1]

        for seg_idx, segment in enumerate(segments):
            end_time = segment["end"]
            end_frame = int(end_time * 25)
            start_frame = max(0, end_frame - clip_size)
            # Clamp to available data
            start_frame = min(start_frame, max(0, total_data_frames - clip_size))

            x = video_data['img'].to(device, non_blocking=True)[:, start_frame:start_frame + clip_size]
            listener_ref = video_data['listener_video_clip'].to(device, non_blocking=True)[:, start_frame:start_frame + clip_size]
            speaker_Audio = video_data['speaker_audio_clip'].to(device, non_blocking=True)[:, start_frame:start_frame + clip_size]
            listener_video_clip_last = video_data['listener_video_clip_last'].to(device, non_blocking=True)[:, start_frame:start_frame + clip_size]
            emo_label = video_data['listener_emotion'].to(device, non_blocking=True)[:, start_frame:start_frame + clip_size]
            speaker_emo = video_data['speaker_emotion'].to(device, non_blocking=True)[:, start_frame:start_frame + clip_size]

            # Pad clips that are shorter than clip_size
            act_len = x.shape[1]
            if act_len < clip_size:
                pad_len = clip_size - act_len
                if act_len > 0:
                    x = torch.cat([x, x[:, -1:].repeat(1, pad_len, 1, 1, 1)], dim=1)
                    listener_ref = torch.cat([listener_ref, listener_ref[:, -1:].repeat(1, pad_len, 1, 1, 1)], dim=1)
                    listener_video_clip_last = torch.cat([listener_video_clip_last, listener_video_clip_last[:, -1:].repeat(1, pad_len, 1, 1, 1)], dim=1)
                    speaker_Audio = torch.cat([speaker_Audio, speaker_Audio[:, -1:].repeat(1, pad_len, 1)], dim=1)
                    emo_label = torch.cat([emo_label, emo_label[:, -1:].repeat(1, pad_len, 1)], dim=1)
                    speaker_emo = torch.cat([speaker_emo, speaker_emo[:, -1:].repeat(1, pad_len, 1)], dim=1)
                else:
                    print(f"Warning: Segment {seg_idx} has 0 frames, skipping.")
                    continue

            x = torch.flip(x, dims=(0,))
            if using_cfg:
                x_uncon = torch.zeros_like(x)
                listener_ref_uncon = torch.zeros_like(listener_ref)
                x = torch.cat([x, x_uncon], dim=0)
                listener_ref = torch.cat([listener_ref, listener_ref_uncon], dim=0)

            if args.sample_method == 'ddim':
                samples = diffusion.ddim_sample_loop(
                    sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device,
                    original_x=x, listener_ref=listener_ref, listener_video_clip_last=listener_video_clip_last,
                    sp_audio_clip=speaker_Audio
                )
            elif args.sample_method == 'ddpm':
                samples = diffusion.p_sample_loop(
                    sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device,
                    original_x=x, listener_ref=listener_ref, listener_video_clip_last=listener_video_clip_last,
                    sp_audio_clip=speaker_Audio
                )

            print(samples.shape)
            if args.use_fp16:
                samples = samples.to(dtype=torch.float16)
            if using_cfg:
                samples = samples[:len(x) // 2]
            b, f, c, h, w = samples.shape

            decoded_batch = []
            for batch_idx in range(b):
                sample_instance = samples[batch_idx]
                decoded_instance = reaction_vae.decode(sample_instance / 0.18215).sample
                decoded_batch.append(decoded_instance)
            samples = torch.stack(decoded_batch, dim=0)  # [b, clip_size, C, H, W]

            seg_samples_list.append(samples)
            emo_label_list.append(emo_label)
            # Undo CFG doubling so we store [b, clip_size, C, H, W]
            _lr = video_data['listener_video_clip'].to(device, non_blocking=True)[:, start_frame:start_frame + clip_size]
            listener_ref_list.append(_lr)
            speaker_emo_list.append(speaker_emo)

        if len(seg_samples_list) == 0:
            print(f"No valid segments at step {step}, skipping.")
            continue

        print(f"[Time] Step {step} | Diffusion sampling ({len(seg_samples_list)} segs): {time.time()-_t_diffusion:.2f}s")
        # Concatenate all segment clips along the time dimension
        out_samples = torch.cat(seg_samples_list, dim=1)  # [b, N_segs*clip_size, C, H, W]
        emo_label = emo_label_list[-1]
        listener_ref = listener_ref_list[-1]
        speaker_emo = speaker_emo_list[-1]
        b = out_samples.shape[0]
        valid_segments = segments[:len(seg_samples_list)]
        # Concatenate listener_ref across all segments so emo evaluation covers
        # the same total frame count as the generated all_face_images.
        # pdb.set_trace()
        listener_ref_all = torch.cat(listener_ref_list, dim=1)  # [b, N_segs*clip_size, C, H, W]

        # Save preview of last generated clip
        video_ = ((seg_samples_list[-1][0] * 0.5 + 0.5) * 255).add_(0.5).clamp_(0, 255).to(dtype=torch.uint8).cpu().permute(0, 2, 3, 1).contiguous()
        video_save_path = os.path.join(args.save_video_path, 'sample_' + str(step) + '.mp4')
        print(video_save_path)
        imageio.mimwrite(video_save_path, video_, fps=8, quality=9)
        print('save path {}'.format(args.save_video_path))

        # ── Call talkinghead ONCE with all segments stacked + segment metadata ──
        # The function uses segment_list to:
        #   1. Run LLM reply per segment (rolling dialogue)
        #   2. Build timeline-aware audio (silence during speaker, TTS in gaps)
        #   3. Fuse the talking-head video over the full stacked clip
        all_face_images = []
        dim_out = out_samples.shape
        _t_talkinghead = time.time()
        _step_scores = {'realism': [], 'bleu': [], 'bert': [], 'mos': [], 'sentiment_emb': []}
        for batch_index in range(len(out_samples)):
            batched_faces, scores, _listener_audio_path = openai_deepseek_talkinghead_genall(
                audio_file, listener_audio_file,
                whisper_model, audio_processor,
                video_data['speaker_audio_path'][batch_index],
                video_data['listener_audio_path'][batch_index],
                emo_label, out_samples[batch_index], pipe, vae, unet, pe,
                batch_index=len(out_samples) * step + batch_index,
                num_frame_sp=dim_out[1],
                segment_list=valid_segments
            )
            all_face_images += batched_faces
            all_score['all_realism_score'] += scores['realism_score']
            all_score['all_sentiment_emb'][batch_size * step + batch_index] = scores['sentiment_emb']
            all_score['all_bleu_scores'] += scores['bleu_scores']
            all_score['all_bert_scores'] += scores['bert_scores']
            mos_value = scores.get('mos_score')
            if mos_value is not None:
                all_score['all_mos_scores'] += mos_value
            _step_scores['realism'].append(scores['realism_score'])
            _step_scores['bleu'].append(scores['bleu_scores'])
            _step_scores['bert'].append(scores['bert_scores'])
            _step_scores['mos'].append(mos_value)
            _step_scores['sentiment_emb'].append(scores['sentiment_emb'])
            video_ = np.asarray(batched_faces)[:, :, :, ::-1]
            video_save_path = os.path.join(args.save_video_path, 'generated_' + str(len(out_samples) * step + batch_index) + '.mp4')
            print(video_save_path)
            imageio.mimwrite(video_save_path, video_, fps=8, quality=9)
        # pdb.set_trace()
        print(f"[Time] Step {step} | Talking head generation ({len(out_samples)} batch): {time.time()-_t_talkinghead:.2f}s")
        _emo_device = next(encoder_emo.parameters()).device
        all_face_images=torch.tensor(np.transpose(np.asarray(all_face_images)[:,:,:,::-1],(0,3,1,2))/127.5-1,device=_emo_device).float()
        # listener_ref_all: [b, N_segs*clip_size, C, H, W]
        # all_face_images:  [b*T_pred, C, H, W]
        # Pad all_face_images to b*N_segs*clip_size so the reshape at listener_ref_4emo succeeds.
        _target_n = listener_ref_all.shape[0] * listener_ref_all.shape[1]
        if all_face_images.shape[0] < _target_n:
            _pad = _target_n - all_face_images.shape[0]
            all_face_images = torch.cat(
                [all_face_images, all_face_images[-1:].repeat(_pad, 1, 1, 1)], dim=0
            )
            print(f"[Pad] all_face_images {all_face_images.shape[0]-_pad} → {_target_n} frames")
        # #generate talking head
        # result = whisper_model.transcribe(audio_file)
        # print(result["text"])

        # sentences=result["text"].split('.')

        #visualize 
        

        # video_ = ((out_samples[0] * 0.5 + 0.5) * 255).add_(0.5).clamp_(0, 255).to(dtype=torch.uint8).cpu().permute(0, 2, 3, 1).contiguous()
        # video_save_path = os.path.join(args.save_video_path, 'sample_test_ddpm' + str(step)+'.mp4')
        # print(video_save_path)
        # imageio.mimwrite(video_save_path, video_, fps=8, quality=9)
        # print('save path {}'.format(args.save_video_path))
        # pdb.set_trace()


        # #visualize 
        # video_ = ((video_data['img'][0] * 0.5 + 0.5) * 255).add_(0.5).clamp_(0, 255).to(dtype=torch.uint8).cpu().permute(0, 2, 3, 1).contiguous()
        # video_save_path = os.path.join(args.save_video_path, 'original' + str(step)+'.mp4')
        # print(video_save_path)
        # imageio.mimwrite(video_save_path, video_, fps=8, quality=9)
        # print('save path {}'.format(args.save_video_path))

        # video_ = ((out_samples[0] * 0.5 + 0.5) * 255).add_(0.5).clamp_(0, 255).to(dtype=torch.uint8).cpu().permute(0, 2, 3, 1).contiguous()
        # video_save_path = os.path.join(args.save_video_path, 'sample_test_ddpm' + str(step)+'.mp4')
        # print(video_save_path)
        # imageio.mimwrite(video_save_path, video_, fps=8, quality=9)
        # print('save path {}'.format(args.save_video_path))
        # pdb.set_trace()

        # pdb.set_trace()
        #get emo feature
        _t_emo_pred = time.time()
        id_feats,emo_fea,au_disfa,au_bp4d=encoder_emo.forward_listener(all_face_images.to(_emo_device))
        # id_feats,emo_fea,au_disfa,au_bp4d=model.facial_condition.forward_listener(all_face_images,detect_face=True)
        # id_feats,emo_fea,au_disfa,au_bp4d=model.facial_condition.forward_listener(samples)
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
        current_emo=emo#torch.cat([emo,au],dim=-1)
        # current_emo=torch.cat([emo,au],dim=-1)

        # pdb.set_trace()
        # #get emo label from listener image
        # if all_face_images.shape[0]!=listener_ref.shape[0]:
        #     listener_emotion_pred_list.append(current_emo.cpu())
        #     listener_emotion_gt_list.append(emo_label.cpu())
        #     speaker_emotion_gt_list.append(speaker_emo)
        #     continue
        print(f"[Time] Step {step} | Emotion extraction (generated): {time.time()-_t_emo_pred:.2f}s")
        listener_ref_4emo=listener_ref_all.reshape(all_face_images.shape[0],all_face_images.shape[1],all_face_images.shape[2],all_face_images.shape[3])
        _t_emo_gt = time.time()
        id_feats,emo_fea,au_disfa,au_bp4d=encoder_emo.forward_listener(listener_ref_4emo.to(_emo_device))
        
        au=torch.cat([au_bp4d, au_disfa[:, 4:5], au_disfa[:,-2:]],dim=-1)  #[8*b,15]->[b,8,15]
        au=torch.reshape(au,[b,-1,15])

        #
        expression=emo_fea['expression']
        expression_soft=Func.softmax(expression,dim=1)
        valence=emo_fea['valence']
        arousal=emo_fea['arousal']
        emo=torch.cat([expression_soft,torch.unsqueeze(valence,dim=-1),torch.unsqueeze(arousal,dim=-1)],dim=-1)
        emo=torch.reshape(emo,[b,-1,10])
        listener_gt_frommodel_emo=emo#torch.cat([emo,au],dim=-1)
        # listener_gt_frommodel_emo=torch.cat([emo,au],dim=-1)

        #
        # pdb.set_trace()
        print(f"[Time] Step {step} | Emotion extraction (GT listener): {time.time()-_t_emo_gt:.2f}s")
        listener_emotion_pred_list.append(current_emo.cpu())
        listener_emotion_gt_list.append(listener_gt_frommodel_emo.cpu())
        # listener_emotion_gt_list.append(emo_label.cpu())
        speaker_emotion_gt_list.append(speaker_emo)
        print(f"[Time] Step {step} | Total step time: {time.time()-_t_step_start:.2f}s")

        # ── Save per-step checkpoint so a later crash doesn't lose this step ──
        torch.save({
            'step': step,
            'listener_emotion_pred': current_emo.cpu(),
            'listener_emotion_gt':   listener_gt_frommodel_emo.cpu(),
            'speaker_emotion_gt':    speaker_emo.cpu(),
            'scores': _step_scores,
        }, _step_ckpt)
        print(f"[Checkpoint] Step {step} saved → {_step_ckpt}")
    # ── After loop: load all per-step checkpoints sorted by step, then cat ──
    _ckpt_files = sorted(glob.glob(os.path.join(args.save_video_path, 'ckpt_step_*.pt')))
    print(f"[Checkpoint] Loading {len(_ckpt_files)} step files for final merge...")
    _pred_parts, _gt_parts, _sp_parts = [], [], []
    # Reset all_score so we rebuild it entirely from checkpoint files
    all_score['all_realism_score'] = 0
    all_score['all_bleu_scores']   = 0
    all_score['all_bert_scores']   = 0
    all_score['all_mos_scores']    = 0
    all_score['all_sentiment_emb'] = torch.zeros([len(dataset), 10])
    for _cf in _ckpt_files:
        _c = torch.load(_cf, map_location='cpu')
        _pred_parts.append(_c['listener_emotion_pred'])
        _gt_parts.append(_c['listener_emotion_gt'])
        _sp_parts.append(_c['speaker_emotion_gt'])
        # Aggregate per-step scores
        _s = _c.get('scores', {})
        all_score['all_realism_score'] += sum(v for v in _s.get('realism', []) if v is not None)
        all_score['all_bleu_scores']   += sum(v for v in _s.get('bleu',    []) if v is not None)
        all_score['all_bert_scores']   += sum(v for v in _s.get('bert',    []) if v is not None)
        for _mv in _s.get('mos', []):
            if _mv is not None:
                all_score['all_mos_scores'] += _mv
        _step_num = _c['step']
        for _bi, _emb in enumerate(_s.get('sentiment_emb', [])):
            if _emb is not None:
                _idx = batch_size * _step_num + _bi
                if _idx < len(all_score['all_sentiment_emb']):
                    all_score['all_sentiment_emb'][_idx] = _emb.squeeze(0) if _emb.dim() > 1 else _emb

    # Pad along dim=1 (time axis) so all steps have the same length before cat
    def _pad_to_len(parts, dim=1):
        max_len = max(t.shape[dim] for t in parts)
        padded = []
        for t in parts:
            cur = t.shape[dim]
            if cur < max_len:
                reps = [1] * t.dim()
                reps[dim] = -(-max_len // cur)   # ceil division
                t = t.repeat(reps).narrow(dim, 0, max_len)
            padded.append(t)
        return padded

    _pred_parts = _pad_to_len(_pred_parts)
    _gt_parts   = _pad_to_len(_gt_parts)
    _sp_parts   = _pad_to_len(_sp_parts)
    pdb.set_trace()
    listener_emotion_pred = torch.cat(_pred_parts, dim=0)
    listener_emotion_gt   = torch.cat(_gt_parts,   dim=0)
    speaker_emotion_gt    = torch.cat(_sp_parts,   dim=0)

    all_listener_emotion_pred_list.append(listener_emotion_pred.unsqueeze(1))
    all_listener_emotion_pred = torch.cat(all_listener_emotion_pred_list, dim=1)
    # if not os.path.exists(args.save_video_path):
    #     os.makedirs(args.save_video_path)

    print("-----------------Evaluating Metric-----------------")
    _t_metric = time.time()

    # p = args.threads
    # for key,value in all_score.items():
    #     print(key,value/len(listener_emotion_pred))
    # pdb.set_trace()
    language_mse=torch.nn.functional.mse_loss(all_score['all_sentiment_emb'][:len(listener_emotion_gt)],torch.mean(listener_emotion_gt[:,:,:10],1))#check listener_emotion_gt.shape
    print('language_mse',language_mse)
    # pdb.set_trace()
    realism_score=all_score['all_realism_score']/len(listener_emotion_gt)
    bleu_scores=all_score['all_bleu_scores']/len(listener_emotion_gt)
    bert_scores=all_score['all_bert_scores']/len(listener_emotion_gt)
    print('realism_score',realism_score)
    print('bleu_scores',bleu_scores)
    print('bert_scores',bert_scores)
    # Compute average MOS score if collected
    # pdb.set_trace()
    if all_score['all_mos_scores'] > 0:
        mos_score = all_score['all_mos_scores']/len(listener_emotion_gt)
        print('mos_score',mos_score)
    #save and load parameter
    torch.save(all_listener_emotion_pred, "./emo_pred.pt")
    torch.save(speaker_emotion_gt, "./speaker_emo_gt.pt")
    torch.save(listener_emotion_gt, "./listener_emo_gt.pt")
    torch.save(all_score['all_sentiment_emb'][:len(listener_emotion_gt)], "./llm_emo_pred.pt")
    pdb.set_trace()
    if is_2025:
        # react2025 metrics: preds [N, 1, T, 10], targets [N, 1, T, 10]
        emo_pred_reshape = all_listener_emotion_pred.cpu()
        listener_emo_gt_reshape = listener_emotion_gt.unsqueeze(1).cpu()
        if len(emo_pred_reshape) < len(dataset):
            repeat_times = len(dataset) // len(emo_pred_reshape)
            fake_emo_pred = emo_pred_reshape.repeat(repeat_times, 1, 1, 1)
            fake_listener_gt = listener_emo_gt_reshape.repeat(repeat_times, 1, 1, 1)
            FRC = compute_FRC_nomp(fake_emo_pred, fake_listener_gt)
            FRD = compute_FRD_nomp(fake_emo_pred, fake_listener_gt)
        else:
            FRC = compute_FRC_nomp(emo_pred_reshape, listener_emo_gt_reshape)
            FRD = compute_FRD_nomp(emo_pred_reshape, listener_emo_gt_reshape)
        FRDvs = compute_FRDvs_2025(emo_pred_reshape)
        FRVar = compute_FRVar_2025(emo_pred_reshape)
        # TLCC = compute_TLCC_nomp(emo_pred_reshape, speaker_emotion_gt.cpu())
        print("FRC:", FRC)
        print("FRD:", FRD)
        print("FRVar:", FRVar)
        print("Metric: | FRC: {:.5f} | FRD: {:.5f} | FRVar: {:.5f} | FRDvs: {:.5f}".format(FRC, FRD, FRVar, FRDvs))
        print("Latex-friendly --> model_name & {:.2f} & {:.2f} & {:.4f} & {:.4f} \\\\".format(FRC, FRD, FRVar, FRDvs))
    else:
        # react2024 metrics
        # TLCC = compute_TLCC(all_listener_emotion_pred.cpu(), speaker_emotion_gt.cpu())
        if len(all_listener_emotion_pred) < len(dataset):
            repeat_times = len(dataset) // len(all_listener_emotion_pred)
            fake_all_listener_emotion_pred = all_listener_emotion_pred.repeat(repeat_times, 1, 1, 1)
            fake_listener_emotion_gt = listener_emotion_gt.repeat(repeat_times, 1, 1)
            FRC = compute_FRC(args, fake_all_listener_emotion_pred.cpu(), fake_listener_emotion_gt.cpu(), val_test='test')
            FRD = compute_FRD(args, fake_all_listener_emotion_pred.cpu(), fake_listener_emotion_gt.cpu(), val_test='test')
        else:
            FRC = compute_FRC(args, all_listener_emotion_pred.cpu(), listener_emotion_gt.cpu(), val_test='test')
            FRD = compute_FRD(args, all_listener_emotion_pred.cpu(), listener_emotion_gt.cpu(), val_test='test')
        FRDvs = compute_FRDvs(all_listener_emotion_pred.cpu())
        FRVar = compute_FRVar(all_listener_emotion_pred.cpu())
        smse = compute_s_mse(all_listener_emotion_pred.cpu())
        print("FRC: %.4f;FRD: %.4f;\n FRDvs: %.4f;FRVar: %.4f;smse: %.4f;" % (FRC, FRD, FRDvs, FRVar, smse))
    print(f"[Time] Metric computation: {time.time()-_t_metric:.2f}s")

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/reaction/reaction_test.yaml")
    
    if 'sbg'  in platform.node() or 'dg' in platform.node() or 'xlg' in platform.node():
        # parser.add_argument("--ckpt", type=str, default="/gpfs/scratch/acw569/checkpoint/latte_check_new/021-Latte-B-4-F64S3-reaction-Amp/checkpoints/0060000.pt")

        parser.add_argument("--ckpt", type=str, default="/gpfs/scratch/acw569/checkpoint/latte_check_new/046-Latte-S-2-F64S3-reaction-Amp/checkpoints/0050000.pt")
    else:
        parser.add_argument("--ckpt", type=str, default="/data/szl/code/latte_reaction_new/latte_reaction_check/023-Latte-B-4-F64S3-reaction-Amp/checkpoints/0260000.pt")
    # parser.add_argument("--ckpt", type=str, default="/import/nobackup_mmv_ioannisp/zs003/code/face_reaction/latte_reaction/results/123-Latte-S-2-F32S3-reaction-Amp/checkpoints/0210000.pt")
    #/import/nobackup_mmv_ioannisp/zs003/code/face_reaction/latte_reaction/checkpoint_fromhpc/0250000.pt
    
    #wrong label:/import/nobackup_mmv_ioannisp/zs003/code/face_reaction/latte_reaction/results/092-Latte-S-2-F32S3-reaction-Amp/checkpoints/0240000.pt
    parser.add_argument("--save_video_path", type=str, default="./sample_videos/")
    parser.add_argument("--api_key", type=str, default="",
                        help="OpenAI-compatible API key (e.g. your ChatAnywhere key)")
    parser.add_argument("--api_base_url", type=str, default="https://api.chatanywhere.tech/v1",
                        help="Base URL for the OpenAI-compatible API endpoint")
    args = parser.parse_args()
    # Expose key/URL as env vars so evaluate_LLM can read them without threading args through every call
    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key
    os.environ["OPENAI_BASE_URL"] = args.api_base_url
    omega_conf = OmegaConf.load(args.config)
    omega_conf.ckpt = args.ckpt
    omega_conf.save_video_path = args.save_video_path
    main(omega_conf)
