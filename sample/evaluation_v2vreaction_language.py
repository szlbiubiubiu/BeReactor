# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# deepseek vllm :vllm serve "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" --max_model 4096 --port 8000 --tensor-parallel-size 2
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


from models.model_irse import IDnet
from models.basenet import LNDnet
from models.seg_model_2 import BGnet
from models.emo_net import EmoNet
from models.emo_fr_net.au_model import MEFL
import torch.nn.functional as Func
from metric import *   #import evaluation metric
import pdb
import numpy as np
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


#talking head libs
# sys.path.append('/import/nobackup_mmv_ioannisp/zs003/code/face_reaction/latte_reaction/talkinghead/muse')
from talkinghead.muse.musetalk.utils.utils import get_file_type,get_video_fps,datagen,datagen_reaction
from talkinghead.muse.musetalk.utils.preprocessing import get_landmark_and_bbox,read_imgs,coord_placeholder
from talkinghead.muse.musetalk.utils.blending import get_image
from talkinghead.muse.musetalk.utils.utils import load_all_model

from transformers import pipeline
import whisper
import pdb

import edge_tts
import cv2
from tqdm import tqdm
import copy
import torch.nn.functional as F
import re

import platform
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

if 'sbg'  in platform.node() or 'dg' in platform.node():
    pretrain_root=''
else:
    pretrain_root='/homes/zs003/paper_face/face_react/diff/reaction_generation/'
encoder_emo=DiffusionVideoAutoencoder(pretrain_root+"./pretrained_models/model_ir_se50.pth", pretrain_root+"./pretrained_models/mobilenet_224_model_best_gdconv_external.pth.tar", pretrain_root+"./pretrained_models/79999_iter.pth"
                    ,emotion_file_path=pretrain_root+'./pretrained_models/emonet_8.pth', 
                    au_file_path_disfa=pretrain_root+'./pretrained_models/MEFARG_swin_base_DISFA_fold1.pth',
                    au_file_path_bp4d=pretrain_root+'./pretrained_models/MEFARG_swin_base_BP4D_fold1.pth').to(torch.device('cuda:1'))#.cuda()



from openai import OpenAI
import openai

import nltk
nltk.download('punkt_tab')
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
    # sk-6L14dOtlYeIpy6Bxa7MOxYyQ1JmOSfpMyTJrg5lHkL5WEgO2
    client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key="sk-6L14dOtlYeIpy6Bxa7MOxYyQ1JmOSfpMyTJrg5lHkL5WEgO2",#30 yuan
    base_url="https://api.chatanywhere.tech/v1"
    # base_url="https://api.chatanywhere.org/v1"
    )
    # client = OpenAI(
    # # defaults to os.environ.get("OPENAI_API_KEY")
    # api_key="sk-RNBcv6ApKlg8UoCcdP9RtfXQLPEVfdebSB7gnvUi3NlWChUa", #free
    # base_url="https://api.chatanywhere.tech/v1"
    # # base_url="https://api.chatanywhere.org/v1"
    # )
    response = client.chat.completions.create(
            # model="gpt-3.5-turbo",  # or your preferred model
            # model="gpt-4o-mini-ca",
            model="gpt-3.5-turbo-ca",
            
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
        
    # pdb.set_trace()
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

def openai_deepseek_talkinghead_genall(audio_file,listener_audio_file,whisper_model,audio_processor,
    speaker_audio_path,listener_audio_path, emo_label,input_latent, pipe,vae, unet, pe,batch_index=0,batch_size=8,fps=25,
        result_img_save_path='./temp',VOICE = "en-GB-SoniaNeural",return_image4emotion=True,num_frame_sp=32):
    # pdb.set_trace()
    # for step,clip_generated in enumerate(input_latent):

    id_feats,emo_fea,au_disfa,au_bp4d=encoder_emo.forward_listener(input_latent.to(torch.device('cuda:1')))#.to(torch.device('cuda:0'))
    # pdb.set_trace()
    expression=emo_fea['expression'].to(torch.device('cuda:0'))
    expression_soft_sing=F.softmax(expression,dim=1)
    valence_sing=emo_fea['valence'].to(torch.device('cuda:0'))
    arousal_sing=emo_fea['arousal'].to(torch.device('cuda:0'))
    # if step==0:
    expression_soft=expression_soft_sing
    valence=valence_sing
    arousal=arousal_sing
        # else:
        #     expression_soft=torch.cat((expression_soft,expression_soft_sing),dim=0)
        #     valence=torch.cat((valence,valence_sing),dim=0)
        #     arousal=torch.cat((arousal,arousal_sing),dim=0)

    # pdb.set_trace()
    result = whisper_model.transcribe(speaker_audio_path)
    print(result["text"])
    
    sentences=result["text"].split('.')

    listener_result = whisper_model.transcribe(listener_audio_path)#audio_file,listener_audio_path
    # print(result["text"])
        
    listener_reference_sentences=listener_result["text"].split('.')

    # pdb.set_trace()
    # bleu_scores = calculate_bleu_batch([listener_result["text"]], [result["text"]])
    # rouge_scores = calculate_rouge_batch([listener_result["text"]], [result["text"]])
    # bert_scores = calculate_bertscore_batch([listener_result["text"]], [result["text"]])



    rand_num=torch.randint(0,100,(1,))
    all_text=''
    conversation_history = f"system: You are a person with a specific personality, subjected to the random seed:."+str(5)+". reply within 30 words(at least 6 words)\n"
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
    openai_api_key = "None"
    openai_api_base = "http://0.0.0.0:8000/v1"

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    dialogue=[]
    random_number=np.random.randint(low=0, high=100)
    
    for i in range(len(sentences)):
        if i>=5:
            break
        if i==0:
            if len(sentences)!=1:
                chat = [
                {"role": "system", "content": "We are having an interview, pretend you are a person with a specific personality, subject to the random seed:"+str(random_number)\
                +".Please stay tune for more information if the user input is not clear. Please generate an appropriate response within 20 words in each reply, each response should be different from previous reply"},
                {"role": "user", "content": sentences[0]+". (Please reply with a tone of the following emotion degree:\
            Neutral:{}, Happy:{}, 'Sad':{}, 'Surprise':{}, 'Fear':{}, 'Disgust':{}, 'Anger':{}, 'Contempt':{}, 'valence':{}, 'arousal':{}. For example Happy=0 means not happy at all, Happy=1 means extremely happy".format(
                            expression_soft[:,0].mean(),expression_soft[:,1].mean(),expression_soft[:,2].mean(),expression_soft[:,3].mean(),expression_soft[:,4].mean(),
                            expression_soft[:,5].mean(),expression_soft[:,6].mean(),expression_soft[:,7].mean(),valence.mean(),arousal.mean())
                        +'. Remove the emotion number in your reply.)'}
                ]
            else:
                chat = [
                {"role": "system", "content": "We are having an interview, pretend you are a person with a specific personality, subject to the random seed:"+str(random_number)\
                +".Please stay tune for more information if the user input is not clear. Please generate an appropriate response within 50 words(more than 15 words), don't generate repeat content"},
                {"role": "user", "content": sentences[0]+". (Please reply with a tone of the following emotion degree:\
            Neutral:{}, Happy:{}, 'Sad':{}, 'Surprise':{}, 'Fear':{}, 'Disgust':{}, 'Anger':{}, 'Contempt':{}, 'valence':{}, 'arousal':{}. For example Happy=0 means not happy at all, Happy=1 means extremely happy".format(
                            expression_soft[:,0].mean(),expression_soft[:,1].mean(),expression_soft[:,2].mean(),expression_soft[:,3].mean(),expression_soft[:,4].mean(),
                            expression_soft[:,5].mean(),expression_soft[:,6].mean(),expression_soft[:,7].mean(),valence.mean(),arousal.mean())
                        +'. Remove the emotion number in your reply.)'}
                ]
            # {"role": "system", "content": 
            # "Please reply with a tone of emotion level:\
            # Neutral:{single_express[0]}, Happy:{single_express[1]}, 'Sad':{single_express[2]}, 'Surprise':{single_express[3]},\
            #                     'Fear':{single_express[4]}, 'Disgust':{single_express[5]}, 'Anger':{single_express[6]}, 'Contempt':{single_express[7]}, \
            #             'valence':{single_val}, 'arousal':{single_arousal}. For example Happy=0 means not happy at all, Happy=1 means extremely happy"},            
            
            
            
            
            # dialogue=[{"role": "system", "content": "We are having an interview, pretend you are a person with a specific personality, subject to the random seed:"+str(5)\
            # +".Please stay tune for more information if the user input is not clear. Please generate an appropriate response within 10 words in each reply"},
            
            # {"role": "user", "content": sentences[0]}]
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
                        +'. Remove the emotion number in your reply.)'},#+". please reply within 10 words"}
            # <| Assistant |><think>\n\n</think>\n\n
            # chat.append(
            # {"role": "Assistant", "content": '<think>\n\n</think>\n\n'}
            )
            # dialogue.append(
            # {"role": "Assistant", "content": current_text})
            # dialogue.append({"role": "user", "content": sentences[i]})
        
        
        
        response = pipe(chat, max_new_tokens=512)#,pad_token_id=pipe.tokenizer.eos_token_id
        #below is llama3
        # pdb.set_trace()
        current_text=response[0]['generated_text'][-1]['content']
        # pdb.set_trace()

        # #deepseek with thinking tag
        # response = client.chat.completions.create(
        #     model="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        #     messages=chat,
        #     temperature=0.2,
        # )
        # current_text=response.choices[0].message.content.split('\n</think>\n\n')#[1]
        # if len(current_text)==1:
        #     current_tex=current_text#[0]
        # else:   
        #     current_text=current_text[1]
        

        all_text=all_text+current_text
        print(current_text)
    #add final text
    chat.append(
            {"role": "Assistant", "content": current_text})
    realism_score,sentiment_emb=evaluate_LLM(dialogue)
    # all_text
    bleu_scores = calculate_bleu_batch([listener_result["text"]], [all_text])
    rouge_scores = calculate_rouge_batch([listener_result["text"]], [all_text])
    bert_scores = calculate_bertscore_batch([listener_result["text"]], [all_text])
    evaluation_score=dict()
    evaluation_score['realism_score']=realism_score
    evaluation_score['sentiment_emb']=sentiment_emb
    # evaluation_score['valence_score']=valence_score
    # evaluation_score['arousal_score']=arousal_score
    evaluation_score['bleu_scores']=bleu_scores[0]
    # evaluation_score['rouge_scores']=rouge_scores
    evaluation_score['bert_scores']=bert_scores[0]
    # pdb.set_trace()
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
    # pdb.set_trace()
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

    # pdb.set_trace()
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
    # pdb.set_trace()
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
    # if len(input_latent_list_cycle)==0:
    #     pdb.set_trace()
    #     a=1
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


    # pdb.set_trace()
    if return_image4emotion:
        interval_listener=len(res_frame_list)//num_frame_sp
        if interval_listener==0:
            out_images=res_frame_list
        else:
            out_images=res_frame_list[::interval_listener]
        return out_images[:num_frame_sp],evaluation_score
    print("pad talking image to original video")
    pdb.set_trace()
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
    pdb.set_trace()
    cmd_img2video = f"ffmpeg -y -v warning -r {fps} -f image2 -i {result_img_save_path}/%08d.png -vcodec libx264 -vf format=rgb24,scale=out_color_matrix=bt709,format=yuv420p -crf 18 temp.mp4"
    print(cmd_img2video)
    os.system(cmd_img2video)
    

    output_vid_name = os.path.join('./result_talking/siyang',"siyang.mp4")
    cmd_combine_audio = f"ffmpeg -y -v warning -i {OUTPUT_audio_FILE} -i temp.mp4 {output_vid_name}"
    print(cmd_combine_audio)
    os.system(cmd_combine_audio)
    
    os.remove("temp.mp4")
    # # shutil.rmtree(result_img_save_path)
    print(f"result is save to {output_vid_name}")      
    pdb.set_trace()
    # if return_image4emotion:
    #return emotion image
    interval_listener=len(res_frame_list)//num_frame_sp
    out_images=res_frame_list[::interval_listener]
    return out_images[:num_frame_sp],evaluation_score 
def openai_deepseek_talkinghead_genall_llmonly(audio_file,listener_audio_file,whisper_model,audio_processor,
    speaker_audio_path,listener_audio_path, emo_label,input_latent, pipe,vae, unet, pe,batch_index=0,batch_size=8,fps=25,
        result_img_save_path='./temp',VOICE = "en-GB-SoniaNeural",return_image4emotion=True,num_frame_sp=32):
    # pdb.set_trace()
    # for step,clip_generated in enumerate(input_latent):

    id_feats,emo_fea,au_disfa,au_bp4d=encoder_emo.forward_listener(input_latent.to(torch.device('cuda:1')))#.to(torch.device('cuda:0'))
    # pdb.set_trace()
    expression=emo_fea['expression'].to(torch.device('cuda:0'))
    expression_soft_sing=F.softmax(expression,dim=1)
    valence_sing=emo_fea['valence'].to(torch.device('cuda:0'))
    arousal_sing=emo_fea['arousal'].to(torch.device('cuda:0'))
    # if step==0:
    expression_soft=expression_soft_sing
    valence=valence_sing
    arousal=arousal_sing
        # else:
        #     expression_soft=torch.cat((expression_soft,expression_soft_sing),dim=0)
        #     valence=torch.cat((valence,valence_sing),dim=0)
        #     arousal=torch.cat((arousal,arousal_sing),dim=0)

    # pdb.set_trace()
    result = whisper_model.transcribe(speaker_audio_path)
    print(result["text"])
    
    sentences=result["text"].split('.')

    listener_result = whisper_model.transcribe(listener_audio_path)#audio_file,listener_audio_path
    # print(result["text"])
        
    listener_reference_sentences=listener_result["text"].split('.')

    # pdb.set_trace()
    # bleu_scores = calculate_bleu_batch([listener_result["text"]], [result["text"]])
    # rouge_scores = calculate_rouge_batch([listener_result["text"]], [result["text"]])
    # bert_scores = calculate_bertscore_batch([listener_result["text"]], [result["text"]])



    rand_num=torch.randint(0,100,(1,))
    all_text=''
    conversation_history = f"system: You are a person with a specific personality, subjected to the random seed:."+str(5)+". reply within 30 words(at least 6 words)\n"
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
    openai_api_key = "None"
    openai_api_base = "http://0.0.0.0:8000/v1"

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    dialogue=[]
    random_number=np.random.randint(low=0, high=100)
    
    for i in range(len(sentences)):
        if i>=5:
            break
        if i==0:
            if len(sentences)!=1:
                chat = [
                {"role": "system", "content": "We are having an interview, pretend you are a person with a specific personality, subject to the random seed:"+str(random_number)\
                +".Please stay tune for more information if the user input is not clear. Please generate an appropriate response within 10 words in each reply"},
                {"role": "user", "content": sentences[0]+". (Please reply with a tone of the following emotion degree:\
            Neutral:{}, Happy:{}, 'Sad':{}, 'Surprise':{}, 'Fear':{}, 'Disgust':{}, 'Anger':{}, 'Contempt':{}, 'valence':{}, 'arousal':{}. For example Happy=0 means not happy at all, Happy=1 means extremely happy".format(
                            expression_soft[:,0].mean(),expression_soft[:,1].mean(),expression_soft[:,2].mean(),expression_soft[:,3].mean(),expression_soft[:,4].mean(),
                            expression_soft[:,5].mean(),expression_soft[:,6].mean(),expression_soft[:,7].mean(),valence.mean(),arousal.mean())
                        +'. Remove the emotion number in your reply.)'}
                ]
            else:
                chat = [
                {"role": "system", "content": "We are having an interview, pretend you are a person with a specific personality, subject to the random seed:"+str(random_number)\
                +".Please stay tune for more information if the user input is not clear. Please generate an appropriate response within 50 words(more than 15 words)"},
                {"role": "user", "content": sentences[0]+". (Please reply with a tone of the following emotion degree:\
            Neutral:{}, Happy:{}, 'Sad':{}, 'Surprise':{}, 'Fear':{}, 'Disgust':{}, 'Anger':{}, 'Contempt':{}, 'valence':{}, 'arousal':{}. For example Happy=0 means not happy at all, Happy=1 means extremely happy".format(
                            expression_soft[:,0].mean(),expression_soft[:,1].mean(),expression_soft[:,2].mean(),expression_soft[:,3].mean(),expression_soft[:,4].mean(),
                            expression_soft[:,5].mean(),expression_soft[:,6].mean(),expression_soft[:,7].mean(),valence.mean(),arousal.mean())
                        +'. Remove the emotion number in your reply.)'}
                ]
            # {"role": "system", "content": 
            # "Please reply with a tone of emotion level:\
            # Neutral:{single_express[0]}, Happy:{single_express[1]}, 'Sad':{single_express[2]}, 'Surprise':{single_express[3]},\
            #                     'Fear':{single_express[4]}, 'Disgust':{single_express[5]}, 'Anger':{single_express[6]}, 'Contempt':{single_express[7]}, \
            #             'valence':{single_val}, 'arousal':{single_arousal}. For example Happy=0 means not happy at all, Happy=1 means extremely happy"},            
            
            
            
            
            # dialogue=[{"role": "system", "content": "We are having an interview, pretend you are a person with a specific personality, subject to the random seed:"+str(5)\
            # +".Please stay tune for more information if the user input is not clear. Please generate an appropriate response within 10 words in each reply"},
            
            # {"role": "user", "content": sentences[0]}]
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
                        +'. Remove the emotion number in your reply.)'},#+". please reply within 10 words"}
            # <| Assistant |><think>\n\n</think>\n\n
            # chat.append(
            # {"role": "Assistant", "content": '<think>\n\n</think>\n\n'}
            )
            # dialogue.append(
            # {"role": "Assistant", "content": current_text})
            # dialogue.append({"role": "user", "content": sentences[i]})
        
        
        
        response = pipe(chat, max_new_tokens=512)#,pad_token_id=pipe.tokenizer.eos_token_id
        #below is llama3
        # pdb.set_trace()
        current_text=response[0]['generated_text'][-1]['content']
        # pdb.set_trace()

        # #deepseek with thinking tag
        # response = client.chat.completions.create(
        #     model="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        #     messages=chat,
        #     temperature=0.2,
        # )
        # current_text=response.choices[0].message.content.split('\n</think>\n\n')#[1]
        # if len(current_text)==1:
        #     current_tex=current_text#[0]
        # else:   
        #     current_text=current_text[1]
        

        all_text=all_text+current_text
        print(current_text)
    #add final text
    chat.append(
            {"role": "Assistant", "content": current_text})
    realism_score,sentiment_emb=evaluate_LLM(dialogue)
    # all_text
    bleu_scores = calculate_bleu_batch([listener_result["text"]], [all_text])
    rouge_scores = calculate_rouge_batch([listener_result["text"]], [all_text])
    bert_scores = calculate_bertscore_batch([listener_result["text"]], [all_text])
    evaluation_score=dict()
    evaluation_score['realism_score']=realism_score
    evaluation_score['sentiment_emb']=sentiment_emb
    # evaluation_score['valence_score']=valence_score
    # evaluation_score['arousal_score']=arousal_score
    evaluation_score['bleu_scores']=bleu_scores[0]
    # evaluation_score['rouge_scores']=rouge_scores
    evaluation_score['bert_scores']=bert_scores[0]
    
    # # pdb.set_trace()
    # if return_image4emotion:
    #     interval_listener=len(res_frame_list)//num_frame_sp
    #     if interval_listener==0:
    #         out_images=res_frame_list
    #     else:
    #         out_images=res_frame_list[::interval_listener]
    #     return out_images[:num_frame_sp],evaluation_score
    num_frame_sp=32
    return input_latent[:num_frame_sp],evaluation_score 

def calculate_talkinghead_genall(audio_file,whisper_model,audio_processor,emo_label,input_latent, pipe,vae, unet, pe,batch_index=0,batch_size=8,fps=25,
        result_img_save_path='./temp',VOICE = "en-GB-SoniaNeural"):
    # model = whisper.load_model("base.en")

    # pdb.set_trace()
    #get emotion from images
    id_feats,emo_fea,au_disfa,au_bp4d=encoder_emo.forward_listener(input_latent.to(torch.device('cuda:1'))).to(torch.device('cuda:0'))
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
    pdb.set_trace()
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
    pdb.set_trace()
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
    pdb.set_trace()
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
    pdb.set_trace()
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
    pdb.set_trace()
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
    pdb.set_trace()
    cmd_img2video = f"ffmpeg -y -v warning -r {fps} -f image2 -i {result_img_save_path}/%08d.png -vcodec libx264 -vf format=rgb24,scale=out_color_matrix=bt709,format=yuv420p -crf 18 temp.mp4"
    print(cmd_img2video)
    os.system(cmd_img2video)
    

    output_vid_name = os.path.join('./result_talking/siyang',"siyang.mp4")
    cmd_combine_audio = f"ffmpeg -y -v warning -i {OUTPUT_audio_FILE} -i temp.mp4 {output_vid_name}"
    print(cmd_combine_audio)
    os.system(cmd_combine_audio)
    
    os.remove("temp.mp4")
    # # shutil.rmtree(result_img_save_path)
    print(f"result is save to {output_vid_name}")        
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
    clip_size=32 #64
    audio_file='/import/nobackup_mmv_ioannisp/zs003/face_react/new_audio_folder/all_together/NoXI/026_2016-04-06_Nottingham/Expert_video/3.wav'
    listener_audio_file='/import/nobackup_mmv_ioannisp/zs003/face_react/new_audio_folder/all_together/NoXI/026_2016-04-06_Nottingham/Novice_video/3.wav'
    #
    # pdb.set_trace()
    print('check test/val matrix..')
    # if val_test == 'val':
    if 'sbg'  in platform.node() or 'dg' in platform.node():
        args.dataset_path='/data/home/acw569/paper_face/face_react/latte/latte_reaction/val_test_matrix'
    else:
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
    pipe = pipeline("text-generation", "meta-llama/Meta-Llama-3-8B-Instruct", torch_dtype=torch.float16, device=1)

    ### deepseek
    # pipe = pipeline("text-generation", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", torch_dtype=torch.float16, device=1)
    
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

    all_score['all_sentiment_emb']=torch.zeros([len(dataset),10])
    # pdb.set_trace()
    # Labels to condition the model with (feel free to change):
    # double_count=0
    for step, video_data in enumerate(loader):
        print('current step is:{}, batch_size is :{}'.format(step,batch_size))
        # if step<280:
        #     continue
        # Skip steps until we reach the resumed step
        # if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
        #     continue
        # pdb.set_trace()
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


        for index_clip in range((video_data['img'].shape)[1]//clip_size):
            # if len(listener_ref)<=32:
            #     pdb.set_trace()
            #     a=1
            
            x = video_data['img'].to(device, non_blocking=True)[:,index_clip*clip_size:(index_clip+1)*clip_size]#.half()
            listener_ref=video_data['listener_video_clip'].to(device, non_blocking=True)[:,index_clip*clip_size:(index_clip+1)*clip_size]#.half()
            if listener_ref.shape[1]<32:
                pdb.set_trace()
                a=1
            # pdb.set_trace()
            emo_label=video_data['listener_emotion'].to(device, non_blocking=True)[:,index_clip*clip_size:(index_clip+1)*clip_size]
            speaker_emo=video_data['speaker_emotion'].to(device, non_blocking=True)[:,index_clip*clip_size:(index_clip+1)*clip_size]
            original_x=x

            #classifier free guidance, change half to zero
            # pdb.set_trace()
            # listener_ref=torch.flip(listener_ref, dims=(0,))
            x=torch.flip(x, dims=(0,))
            if using_cfg:
                x_uncon=torch.zeros_like(x)
                listener_ref_uncon=torch.zeros_like(listener_ref)
                x=torch.cat([x,x_uncon],dim=0)
                listener_ref=torch.cat([listener_ref,listener_ref_uncon],dim=0)
            # # listener_ref=listener_ref
            # # Create sampling noise:
            # if args.use_fp16:
            #     z = torch.randn(batch_size, args.num_frames, 4, latent_size, latent_size, dtype=torch.float16, device=device) # b c f h w
            # else:
            #     z = torch.randn(batch_size, args.num_frames, 4, latent_size, latent_size, device=device)

            # # Setup classifier-free guidance:
            # # z = torch.cat([z, z], 0)
            # if using_cfg:
            #     z = torch.cat([z, z], 0)
            #     y = torch.randint(0, args.num_classes, (1,), device=device)
            #     y_null = torch.tensor([101] * 1, device=device)
            #     y = torch.cat([y, y_null], dim=0)
            #     model_kwargs = dict(y=y, cfg_scale=args.cfg_scale, use_fp16=args.use_fp16)
            #     sample_fn = model.forward_with_cfg
            # else:
            #     sample_fn = model.forward
            #     model_kwargs = dict(y=None, use_fp16=args.use_fp16)
            # # pdb.set_trace()
            # Sample images:
            if args.sample_method == 'ddim':
                samples = diffusion.ddim_sample_loop(
                    sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device,
                    original_x=x,listener_ref=listener_ref,#torch.flip(listener_ref, dims=(0,))
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
            if using_cfg:
                samples=samples[:len(x)//2]
            b, f, c, h, w = samples.shape
            # pdb.set_trace()
            samples = rearrange(samples, 'b f c h w -> (b f) c h w')
            # if index_clip==0:
            #     out_samples=rearrange(samples, '(b f) c h w -> b f c h w', b=b)
            # else:
            #     out_samples=torch.cat((out_samples,rearrange(samples, '(b f) c h w -> b f c h w', b=b)),dim=1)
            # pdb.set_trace()
            samples = reaction_vae.decode(samples / 0.18215).sample
            samples = rearrange(samples, '(b f) c h w -> b f c h w', b=b)#*0.5+0.5
            if index_clip==0:
                out_samples=samples#rearrange(samples, '(b f) c h w -> b f c h w', b=b)
            else:
                out_samples=torch.cat((out_samples,samples),dim=1)
        # #save original video
        # video_ = ((samples[0] * 0.5 + 0.5) * 255).add_(0.5).clamp_(0, 255).to(dtype=torch.uint8).cpu().permute(0, 2, 3, 1).contiguous()
        # video_save_path = os.path.join(args.save_video_path, 'sample' + '.mp4')
        # print(video_save_path)
        # imageio.mimwrite(video_save_path, video_, fps=8, quality=9)
        # print('save path {}'.format(args.save_video_path))


        all_face_images=[]
        #can consider do not use the decode of vae
        dim_out=out_samples.shape
        # all_face_images=openai_deepseek_talkinghead_genall(audio_file,whisper_model,audio_processor,
        #             emo_label,out_samples.reshape(dim_out[0]*dim_out[1],dim_out[2],dim_out[3],dim_out[4]), pipe,vae, unet, pe,num_frame_sp=dim_out[0]*dim_out[1])     #openai,deepseek
        # all_face_images=torch.tensor(np.transpose(np.asarray(all_face_images),(0,3,1,2))/127.5-1,device=out_samples.device).float()
        # pdb.set_trace()
        for batch_index in range(len(out_samples)):
            openai_deepseek_talkinghead_genall_llmonly
            # openai_deepseek_talkinghead_genall
            batched_faces,scores=openai_deepseek_talkinghead_genall_llmonly(audio_file,listener_audio_file,
            whisper_model,audio_processor,video_data['speaker_audio_path'][batch_index],video_data['listener_audio_path'][batch_index],
                    emo_label,out_samples[batch_index], pipe,vae, unet, pe,num_frame_sp=dim_out[1])     #openai,deepseek
           #process audio
            all_score['all_realism_score']+=scores['realism_score']
            all_score['all_sentiment_emb'][batch_size*step+batch_index]=scores['sentiment_emb']
            all_score['all_bleu_scores']+=scores['bleu_scores']
            all_score['all_bert_scores']+=scores['bert_scores']
           
    #        #faces
    #         all_face_images+=batched_faces
    #         video_ = np.asarray(batched_faces)[:,:,:,::-1]
    #         video_save_path = os.path.join(args.save_video_path, 'generated_' + str(len(out_samples)*step+batch_index)+'.mp4')
    #         print(video_save_path)
    #         imageio.mimwrite(video_save_path, video_, fps=8, quality=9)

    #     all_face_images=torch.tensor(np.transpose(np.asarray(all_face_images)[:,:,:,::-1],(0,3,1,2))/127.5-1,device=out_samples.device).float()
    #     # #generate talking head
    #     # result = whisper_model.transcribe(audio_file)
    #     # print(result["text"])

    #     # sentences=result["text"].split('.')

    #     #visualize 
        

    #     # video_ = ((out_samples[0] * 0.5 + 0.5) * 255).add_(0.5).clamp_(0, 255).to(dtype=torch.uint8).cpu().permute(0, 2, 3, 1).contiguous()
    #     # video_save_path = os.path.join(args.save_video_path, 'sample_test_ddpm' + str(step)+'.mp4')
    #     # print(video_save_path)
    #     # imageio.mimwrite(video_save_path, video_, fps=8, quality=9)
    #     # print('save path {}'.format(args.save_video_path))
    #     # pdb.set_trace()


    #     # #visualize 
    #     # video_ = ((video_data['img'][0] * 0.5 + 0.5) * 255).add_(0.5).clamp_(0, 255).to(dtype=torch.uint8).cpu().permute(0, 2, 3, 1).contiguous()
    #     # video_save_path = os.path.join(args.save_video_path, 'original' + str(step)+'.mp4')
    #     # print(video_save_path)
    #     # imageio.mimwrite(video_save_path, video_, fps=8, quality=9)
    #     # print('save path {}'.format(args.save_video_path))

    #     # video_ = ((out_samples[0] * 0.5 + 0.5) * 255).add_(0.5).clamp_(0, 255).to(dtype=torch.uint8).cpu().permute(0, 2, 3, 1).contiguous()
    #     # video_save_path = os.path.join(args.save_video_path, 'sample_test_ddpm' + str(step)+'.mp4')
    #     # print(video_save_path)
    #     # imageio.mimwrite(video_save_path, video_, fps=8, quality=9)
    #     # print('save path {}'.format(args.save_video_path))
    #     # pdb.set_trace()

    #     # pdb.set_trace()
    #     #get emo feature
    #     id_feats,emo_fea,au_disfa,au_bp4d=encoder_emo.forward_listener(all_face_images.to(torch.device('cuda:1')))#.to(torch.device('cuda:0'))
    #     # id_feats,emo_fea,au_disfa,au_bp4d=model.facial_condition.forward_listener(all_face_images,detect_face=True)
    #     # id_feats,emo_fea,au_disfa,au_bp4d=model.facial_condition.forward_listener(samples)
    #     # id=id.reshape(b*v,c,w,h)
    #     # lnd=lnd.reshape(b*v,c_l,w_l,h_l)
        
    #     # # pdb.set_trace()
    #     # self.idnet.eval()
    #     # id_feats = self.idnet(id)  # [B, 512]

    #     #emo_feature
    #     # pdb.set_trace()
    #     au=torch.cat([au_bp4d, au_disfa[:, 4:5], au_disfa[:,-2:]],dim=-1)  #[8*b,15]->[b,8,15]
    #     au=torch.reshape(au,[b,-1,15])

    #     #
    #     expression=emo_fea['expression']
    #     expression_soft=Func.softmax(expression,dim=1)
    #     valence=emo_fea['valence']
    #     arousal=emo_fea['arousal']
    #     emo=torch.cat([expression_soft,torch.unsqueeze(valence,dim=-1),torch.unsqueeze(arousal,dim=-1)],dim=-1)
    #     emo=torch.reshape(emo,[b,-1,10])
    #     current_emo=emo#torch.cat([emo,au],dim=-1)
    #     # current_emo=torch.cat([emo,au],dim=-1)

    #     # pdb.set_trace()
    #     # #get emo label from listener image
    #     # if all_face_images.shape[0]!=listener_ref.shape[0]:
    #     #     listener_emotion_pred_list.append(current_emo.cpu())
    #     #     listener_emotion_gt_list.append(emo_label.cpu())
    #     #     speaker_emotion_gt_list.append(speaker_emo)
    #     #     continue
        # listener_ref_4emo=listener_ref.reshape(all_face_images.shape[0],all_face_images.shape[1],all_face_images.shape[2],all_face_images.shape[3])
        listener_ref_4emo=listener_ref.reshape(dim_out[0]*dim_out[1],3,256,256)

        id_feats,emo_fea,au_disfa,au_bp4d=encoder_emo.forward_listener(listener_ref_4emo.to(torch.device('cuda:1')))#.to(torch.device('cuda:0'))
        
        au=torch.cat([au_bp4d, au_disfa[:, 4:5], au_disfa[:,-2:]],dim=-1)  #[8*b,15]->[b,8,15]
        au=torch.reshape(au,[b,-1,15])

    #     #
        expression=emo_fea['expression']
        expression_soft=Func.softmax(expression,dim=1)
        valence=emo_fea['valence']
        arousal=emo_fea['arousal']
        emo=torch.cat([expression_soft,torch.unsqueeze(valence,dim=-1),torch.unsqueeze(arousal,dim=-1)],dim=-1)
        emo=torch.reshape(emo,[b,-1,10])
        listener_gt_frommodel_emo=emo#torch.cat([emo,au],dim=-1)
        # listener_gt_frommodel_emo=torch.cat([emo,au],dim=-1)

    #     #
    #     # pdb.set_trace()
    #     listener_emotion_pred_list.append(current_emo.cpu())
        listener_emotion_gt_list.append(listener_gt_frommodel_emo.cpu())
    #     # listener_emotion_gt_list.append(emo_label.cpu())
    #     speaker_emotion_gt_list.append(speaker_emo)
        

    # # pdb.set_trace()
    # listener_emotion_pred = torch.cat(listener_emotion_pred_list, dim = 0)
    listener_emotion_gt = torch.cat(listener_emotion_gt_list, dim = 0)

    # speaker_emotion_gt=torch.cat(speaker_emotion_gt_list, dim = 0)

    # all_listener_emotion_pred_list.append(listener_emotion_pred.unsqueeze(1))
    # all_listener_emotion_pred = torch.cat(all_listener_emotion_pred_list, dim=1)
    # # if not os.path.exists(args.save_video_path):
    # #     os.makedirs(args.save_video_path)

    # print("-----------------Evaluating Metric-----------------")

    # # p = args.threads
    # # for key,value in all_score.items():
    # #     print(key,value/len(listener_emotion_pred))
    # # pdb.set_trace()
    language_mse=torch.nn.functional.mse_loss(all_score['all_sentiment_emb'][:len(listener_emotion_gt)],torch.mean(listener_emotion_gt[:,:,:10],1))#check listener_emotion_gt.shape
    print('language_mse',language_mse)
    # pdb.set_trace()
    realism_score=all_score['all_realism_score']/len(listener_emotion_gt)
    bleu_scores=all_score['all_bleu_scores']/len(listener_emotion_gt)
    bert_scores=all_score['all_bert_scores']/len(listener_emotion_gt)
    print('realism_score',realism_score)
    print('bleu_scores',bleu_scores)
    print('bert_scores',bert_scores)
    #save and load parameter
    torch.save(all_listener_emotion_pred, "./emo_pred.pt")
    torch.save(speaker_emotion_gt, "./speaker_emo_gt.pt")
    torch.save(listener_emotion_gt, "./listener_emo_gt.pt")
    torch.save(all_score['all_sentiment_emb'][:len(listener_emotion_gt)], "./llm_emo_pred.pt")
    # # If you have problems running function compute_TLCC_mp, please replace this function with function compute_TLCC
    # # TLCC = compute_TLCC(all_listener_emotion_pred.cpu(), speaker_emotion_gt.cpu())
    # # pdb.set_trace()
    # # If you have problems running function compute_FRC_mp, please replace this function with function compute_FRC
    # if len(all_listener_emotion_pred)<len(dataset):
    #     repeat_times=len(dataset)//len(all_listener_emotion_pred)
    #     fake_all_listener_emotion_pred=all_listener_emotion_pred.repeat(repeat_times,1,1,1)
    #     fake_listener_emotion_gt=listener_emotion_gt.repeat(repeat_times,1,1)
    #     FRC = compute_FRC(args, fake_all_listener_emotion_pred.cpu(), fake_listener_emotion_gt.cpu(), val_test='test')

    #     # If you have problems running function compute_FRD_mp, please replace this function with function compute_FRD
    #     FRD = compute_FRD(args, fake_all_listener_emotion_pred.cpu(), fake_listener_emotion_gt.cpu(), val_test='test')
    # else:
    #     FRC = compute_FRC(args, all_listener_emotion_pred.cpu(), listener_emotion_gt.cpu(), val_test='test')

    #     # If you have problems running function compute_FRD_mp, please replace this function with function compute_FRD
    #     FRD = compute_FRD(args, all_listener_emotion_pred.cpu(), listener_emotion_gt.cpu(), val_test='test')

    # FRDvs = compute_FRDvs(all_listener_emotion_pred.cpu())
    # FRVar  = compute_FRVar(all_listener_emotion_pred.cpu())
    # smse  = compute_s_mse(all_listener_emotion_pred.cpu())
    # # print('TLCC',TLCC)
    # print("TLCC: %.4f; FRC: %.4f;FRD: %.4f;\n FRDvs: %.4f;FRVar: %.4f;smse: %.4f;" % (TLCC, FRC, FRD, FRDvs, FRVar, smse))

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/reaction/reaction_test.yaml")
    
    if 'sbg'  in platform.node() or 'dg' in platform.node():
        parser.add_argument("--ckpt", type=str, default="/data/scratch/acw569/checkpoint/reaction/0250000.pt")
    else:
        parser.add_argument("--ckpt", type=str, default="/import/nobackup_mmv_ioannisp/zs003/code/face_reaction/latte_reaction/checkpoint_fromhpc/0250000.pt")
    # parser.add_argument("--ckpt", type=str, default="/import/nobackup_mmv_ioannisp/zs003/code/face_reaction/latte_reaction/results/123-Latte-S-2-F32S3-reaction-Amp/checkpoints/0210000.pt")
    #/import/nobackup_mmv_ioannisp/zs003/code/face_reaction/latte_reaction/checkpoint_fromhpc/0250000.pt
    
    #wrong label:/import/nobackup_mmv_ioannisp/zs003/code/face_reaction/latte_reaction/results/092-Latte-S-2-F32S3-reaction-Amp/checkpoints/0240000.pt
    parser.add_argument("--save_video_path", type=str, default="./sample_videos/")
    args = parser.parse_args()
    omega_conf = OmegaConf.load(args.config)
    omega_conf.ckpt = args.ckpt
    omega_conf.save_video_path = args.save_video_path
    main(omega_conf)
