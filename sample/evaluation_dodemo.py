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
# from models.clip import TextEmbedder
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
import shutil
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
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
import librosa
import gc


def get_speaker_speaking_intervals(whisper_result, fps=25, total_frames=None):
    """从whisper结果计算speaker说话时间段和停顿时间段
    
    Args:
        whisper_result: whisper转录结果
        fps: 视频帧率
        total_frames: 视频总帧数（用于限制范围）
    
    Returns:
        speaking_intervals: [(start_frame, end_frame), ...] speaker说话的时间段
        pause_intervals: [(start_frame, end_frame), ...] listener可反应的停顿时间段
    """
    speaking_intervals = []
    
    if 'segments' in whisper_result:
        for segment in whisper_result['segments']:
            start_time = segment['start']
            end_time = segment['end']
            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps)
            
            if total_frames is not None:
                end_frame = min(end_frame, total_frames)
            
            speaking_intervals.append((start_frame, end_frame))
    
    # 生成停顿时间段（speaker不说话的时间段）
    pause_intervals = []
    if speaking_intervals:
        # 开头的停顿
        if speaking_intervals[0][0] > 0:
            pause_intervals.append((0, speaking_intervals[0][0]))
        
        # 中间的停顿
        for i in range(len(speaking_intervals) - 1):
            pause_start = speaking_intervals[i][1]
            pause_end = speaking_intervals[i + 1][0]
            if pause_end > pause_start:
                pause_intervals.append((pause_start, pause_end))
        
        # 结尾的停顿
        if total_frames is not None and speaking_intervals[-1][1] < total_frames:
            pause_intervals.append((speaking_intervals[-1][1], total_frames))
    
    return speaking_intervals, pause_intervals


def detect_speech_pauses(whisper_result, audio_file, fps=25, max_pauses=4):
    """检测speech中的停顿点（句子结束、显著停顿等）
    
    Args:
        whisper_result: whisper转录结果
        audio_file: 音频文件路径
        fps: 视频帧率
        max_pauses: 最多检测的停顿点数量 (默认为4)
    
    Returns:
        pause_frames: 停顿发生的帧号列表，最多max_pauses个
    """
    # 从whisper segments提取时间戳
    pause_frames = []
    
    if 'segments' in whisper_result:
        for i, segment in enumerate(whisper_result['segments']):
            # 句子结束时刻 (通常是反应生成的好时机)
            end_time = segment['end']
            end_frame = int(end_time * fps)
            pause_frames.append(end_frame)
            
            # 如果段落间隔>0.5秒，也作为停顿点
            if i < len(whisper_result['segments']) - 1:
                next_start = whisper_result['segments'][i+1]['start']
                gap = next_start - end_time
                if gap > 0.5:
                    pause_frames.append(end_frame + int(gap * fps * 0.5))
    
    pause_frames = sorted(list(set(pause_frames)))
    
    # 限制停顿点数量，最多max_pauses个
    if len(pause_frames) > max_pauses:
        # 均匀分布选择max_pauses个停顿点
        indices = [int(i * len(pause_frames) / max_pauses) for i in range(max_pauses)]
        pause_frames = [pause_frames[idx] for idx in indices]
    
    return pause_frames


def detect_expression_keyframes(speaker_video, fps=25, expression_threshold=0.3):
    """检测speaker面部表情变化较大的关键帧
    
    Args:
        speaker_video: speaker视频 [B, T, C, H, W]
        fps: 帧率
        expression_threshold: 表情变化阈值
    
    Returns:
        keyframe_indices: 关键帧的索引列表
    """
    # 使用pretrained emotion model来检测表情变化
    keyframe_indices = []
    
    # 为了简化，这里采用基于光流或其他简单特征的启发式方法
    # 计算相邻帧的差异
    if len(speaker_video.shape) == 5:  # [B, T, C, H, W]
        video_seq = speaker_video[0]  # 取第一个batch的视频序列
    else:
        video_seq = speaker_video
    
    # 计算相邻帧的L2距离来检测变化
    frame_differences = []
    for t in range(1, video_seq.shape[0]):
        diff = torch.norm(video_seq[t] - video_seq[t-1], p=2).item()
        frame_differences.append(diff)
    
    # 找出变化较大的帧（>阈值*平均变化）
    if frame_differences:
        avg_diff = np.mean(frame_differences)
        threshold = avg_diff * expression_threshold
        
        for t, diff in enumerate(frame_differences):
            if diff > threshold:
                keyframe_indices.append(t + 1)  # +1因为差异是从第1帧开始的
    
    return sorted(keyframe_indices)


def merge_generation_triggers(pause_frames, keyframe_indices, min_interval=16, fps=25):
    """合并停顿点和关键帧，生成最优的反应生成时机
    
    Args:
        pause_frames: 停顿帧号列表
        keyframe_indices: 关键帧索引列表
        min_interval: 最小生成间隔（帧数）
        fps: 帧率
    
    Returns:
        generation_points: 建议的生成时机（帧号）
    """
    # 合并两个信号，优先考虑停顿点，其次是关键帧
    all_triggers = []
    
    # 停顿点权重更高（停顿=更多语义相关的反应时机）
    for frame in pause_frames:
        all_triggers.append((frame, 'pause', 2.0))
    
    # 关键帧权重较低
    for frame in keyframe_indices:
        all_triggers.append((frame, 'keyframe', 1.0))
    
    # 按权重和时间排序
    all_triggers.sort(key=lambda x: (x[0], -x[2]))
    
    # 去除过于接近的触发点（保留权重更高的）
    generation_points = []
    for frame, trigger_type, weight in all_triggers:
        # 检查是否与已选择的点太接近
        too_close = any(abs(frame - gp['frame']) < min_interval for gp in generation_points)
        if not too_close:
            generation_points.append({'frame': frame, 'type': trigger_type, 'weight': weight})
    
    return sorted(generation_points, key=lambda x: x['frame'])


def is_frame_in_interval(frame, intervals):
    """检查帧号是否在指定的时间段内
    
    Args:
        frame: 帧号
        intervals: [(start, end), ...] 时间段列表
    
    Returns:
        True if frame is in any interval
    """
    for start, end in intervals:
        if start <= frame < end:
            return True
    return False


def get_clip_generation_mode(index_clip, clip_size, speaking_intervals, pause_intervals):
    """获取clip的生成模式
    
    Args:
        index_clip: clip索引
        clip_size: 每个clip的帧数
        speaking_intervals: [(start, end), ...] speaker说话的时间段
        pause_intervals: [(start, end), ...] listener可反应的停顿时间段
    
    Returns:
        mode: 'generate' - 生成新反应, 'hold' - 保持前一帧, 'static' - 使用静态帧
        reason: 原因说明
    """
    clip_start = index_clip * clip_size
    clip_end = (index_clip + 1) * clip_size
    clip_mid = (clip_start + clip_end) // 2
    
    # 检查clip是否主要在停顿期间
    in_pause = is_frame_in_interval(clip_mid, pause_intervals)
    in_speaking = is_frame_in_interval(clip_mid, speaking_intervals)
    
    if in_pause:
        return 'generate', f'In pause interval at frame {clip_mid}'
    elif in_speaking:
        return 'hold', f'In speaking interval at frame {clip_mid}, hold last frame'
    else:
        # 检查是否接近任何区间边界
        for start, end in speaking_intervals:
            if abs(clip_mid - start) < clip_size or abs(clip_mid - end) < clip_size:
                return 'generate', f'Near speaking interval boundary'
        
        return 'hold', f'Unassigned area, hold last frame at frame {clip_mid}'


def interpolate_frames_with_transition(generated_frames, reference_frames, transition_ratio=0.75, clip_size=64):
    """对生成帧应用过渡效果：前75%保持生成的反应，后25%逐渐过渡到参考帧
    
    Args:
        generated_frames: 生成的帧 [B, T, C, H, W]
        reference_frames: 参考帧 [B, T, C, H, W] (可选，用于过渡)
        transition_ratio: 生成帧的比例（默认0.75为前75%生成，后25%过渡）
        clip_size: 帧数, 默认64
    
    Returns:
        interpolated_frames: 插值后的帧 [B, T, C, H, W]
    """
    B, T, C, H, W = generated_frames.shape
    interpolated_frames = generated_frames.clone()
    
    # 计算过渡区间
    transition_start = int(T * transition_ratio)
    transition_end = T
    transition_length = transition_end - transition_start
    
    if transition_length > 0 and reference_frames is not None:
        # 在过渡区间内逐渐从生成帧过渡到参考帧
        for frame_idx in range(transition_start, transition_end):
            # 计算过渡权重：从1.0到0.0（生成帧权重逐渐降低）
            transition_progress = (frame_idx - transition_start) / transition_length
            generated_weight = 1.0 - transition_progress  # 生成帧权重从1.0降到0.0
            ref_weight = transition_progress              # 参考帧权重从0.0升到1.0
            
            # 混合生成帧和参考帧
            interpolated_frames[:, frame_idx, :, :, :] = (
                generated_weight * generated_frames[:, frame_idx, :, :, :] +
                ref_weight * reference_frames[:, frame_idx % reference_frames.shape[1], :, :, :]
            )
    elif transition_length > 0:
        # 如果没有参考帧，使用最后一个生成帧作为过渡目标
        last_frame = generated_frames[:, transition_start - 1:transition_start, :, :, :].clone()
        
        for frame_idx in range(transition_start, transition_end):
            transition_progress = (frame_idx - transition_start) / transition_length
            generated_weight = 1.0 - transition_progress
            static_weight = transition_progress
            
            interpolated_frames[:, frame_idx, :, :, :] = (
                generated_weight * generated_frames[:, frame_idx, :, :, :] +
                static_weight * last_frame[:, 0, :, :, :]
            )
    
    return interpolated_frames


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
            b,c,w,h=id.shape
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

# if 'sbg'  in platform.node() or 'dg' in platform.node():
pretrain_root=''
# else:
#     pretrain_root='/homes/zs003/paper_face/face_react/diff/reaction_generation/'
encoder_emo=DiffusionVideoAutoencoder(pretrain_root+"./pretrained_models/model_ir_se50.pth", pretrain_root+"./pretrained_models/mobilenet_224_model_best_gdconv_external.pth.tar", pretrain_root+"./pretrained_models/79999_iter.pth"
                    ,emotion_file_path=pretrain_root+'./pretrained_models/emonet_8.pth', 
                    au_file_path_disfa=pretrain_root+'./pretrained_models/MEFARG_swin_base_DISFA_fold1.pth',
                    au_file_path_bp4d=pretrain_root+'./pretrained_models/MEFARG_swin_base_BP4D_fold1.pth').to(torch.device('cuda:0'))#.cuda()



from openai import OpenAI
import openai

import nltk
# nltk.download('punkt_tab')
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import bert_score

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
audio_device = "cuda:0" if torch.cuda.is_available() else "cpu"

audio_model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-v1").to(audio_device)
tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")

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

def openai_deepseek_talkinghead_genall(whisper_model,audio_processor,
    speaker_audio_path,listener_audio_path, emo_label,input_latent, pipe,vae, unet, pe,batch_index=0,batch_size=8,fps=10,
        result_img_save_path='./temp',VOICE = "en-GB-SoniaNeural",return_image4emotion=False,num_frame_sp=32,num_llm_responses=1):
    # pdb.set_trace()
    # for step,clip_generated in enumerate(input_latent):

    id_feats,emo_fea,au_disfa,au_bp4d=encoder_emo.forward_listener(input_latent.to(torch.device('cuda:0')))#.to(torch.device('cuda:0'))
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
    
    sentences=['Hi there! Nice to Meet you! Please introduce yourself?']#[result["text"]]#.split('.')  for demo, find a speaker that only speak 1 sentence
    # sentences=['Have you traveled to Vancouver? What impresses you most?']
    # pdb.set_trace()
    if len(result["text"].split('.'))==1:
        # pdb.set_trace()
        a=1
    #sentences=result["text"].split('.')
    listener_result = whisper_model.transcribe(listener_audio_path)#audio_file,listener_audio_path
    # print(result["text"])
        
    listener_reference_sentences=listener_result["text"].split('.')

    # pdb.set_trace()
    # bleu_scores = calculate_bleu_batch([listener_result["text"]], [result["text"]])
    # rouge_scores = calculate_rouge_batch([listener_result["text"]], [result["text"]])
    # bert_scores = calculate_bertscore_batch([listener_result["text"]], [result["text"]])



    rand_num=torch.randint(0,100,(1,))
    all_text=''
    conversation_history = f"system: You are a person with a specific personality(a sales), subjected to the random seed:."+str(6)+". reply within 30 words(at least 6 words)\n"
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
    # else:
    tone_pos=1
    # 确保生成至少num_llm_responses个回应
    num_sentences_to_generate = max(len(sentences), num_llm_responses)
    print(f"[LLM] Generating {num_llm_responses} LLM responses (sentences: {len(sentences)})")
    
    # pdb.set_trace()
    for i in range(num_sentences_to_generate):
        if i>=5:  # 最多生成5个回应
            break
        
        # 如果超过句子数量，使用最后一个句子或通用句子
        current_sentence_idx = min(i, len(sentences) - 1)
        current_sentence = sentences[current_sentence_idx]
        
        if i==0:
            if len(sentences)!=1:
            #     chat = [
            #     {"role": "system", "content": "We are having an interview, pretend you are a person with a specific personality, subject to the random seed:"+str(random_number)\
            #     +".Please stay tune for more information if the user input is not clear. Please generate an appropriate response within 10 words in each reply"},
            #     {"role": "user", "content": sentences[0]+". (Please reply with a tone of the following emotion degree:\
            # Neutral:{}, Happy:{}, 'Sad':{}, 'Surprise':{}, 'Fear':{}, 'Disgust':{}, 'Anger':{}, 'Contempt':{}, 'valence':{}, 'arousal':{}. For example Happy=0 means not happy at all, Happy=1 means extremely happy".format(
            #                 expression_soft[:,0].mean(),expression_soft[:,1].mean(),expression_soft[:,2].mean(),expression_soft[:,3].mean(),expression_soft[:,4].mean(),
            #                 expression_soft[:,5].mean(),expression_soft[:,6].mean(),expression_soft[:,7].mean(),valence.mean(),arousal.mean())
            #             +'. Remove the emotion number in your reply.)'}
            #     ]
                chat = [
                {"role": "system", "content": "We are having an interview, pretend you are a person with a specific personality(a sales), subject to the random seed:"+str(random_number)\
                +".Please stay tune for more information if the user input is not clear. Please generate an appropriate response within 50 words in each reply. Return a binary number describing of the response level, 0 means negative and 1 means postive"},
                {"role": "user", "content": current_sentence}
                ]
            else:
                chat = [
                {"role": "system", "content": "We are having an interview, pretend you are a person with a specific personality(a sales), subject to the random seed:"+str(random_number)\
                +".Please stay tune for more information if the user input is not clear. Please generate an appropriate response within 100 words(more than 10 words)"},
                {"role": "user", "content": current_sentence}
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
                {"role": "user", "content": current_sentence+". (Please reply with a tone of the following emotion degree:\
            Neutral:{}, Happy:{}, 'Sad':{}, 'Surprise':{}, 'Fear':{}, 'Disgust':{}, 'Anger':{}, 'Contempt':{}, 'valence':{}, 'arousal':{}. For example Happy=0 means not happy at all, Happy=1 means extremely happy".format(
                            expression_soft[:,0].mean(),expression_soft[:,1].mean(),expression_soft[:,2].mean(),expression_soft[:,3].mean(),expression_soft[:,4].mean(),
                            expression_soft[:,5].mean(),expression_soft[:,6].mean(),expression_soft[:,7].mean(),valence.mean(),arousal.mean())
                        +'. Remove the emotion number in your reply. )'},#+". please reply within 10 words"}
            # <| Assistant |><think>\n\n</think>\n\n
            # chat.append(
            # {"role": "Assistant", "content": '<think>\n\n</think>\n\n'}
            )
            # dialogue.append(
            # {"role": "Assistant", "content": current_text})
            # dialogue.append({"role": "user", "content": current_sentence})
        
        
        
        response = pipe(chat, max_new_tokens=512)#,pad_token_id=pipe.tokenizer.eos_token_id
        #below is llama3
        # pdb.set_trace()
        text_4_split=response[0]['generated_text'][-1]['content'].split('(')
        current_text=text_4_split[0]#response[0]['generated_text'][-1]['content'].split('(')
        if len(text_4_split)==2:
            # number_string = text_4_split[1].strip(')')
            # # Convert the resulting string '1' to an integer
            # tone_pos = int(number_string)
            tone_pos = re.findall(r"\d+\.?\d*",text_4_split[1])[-1]
            # print(re.findall(r"\d+\.?\d*",string))  
            # =
        
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
    # pdb.set_trace()
    print(chat)
    realism_score,sentiment_emb=evaluate_LLM(chat)
    # all_text
    bleu_scores = calculate_bleu_batch([listener_result["text"]], [all_text])
    rouge_scores = calculate_rouge_batch([listener_result["text"]], [all_text])
    bert_scores = calculate_bertscore_batch([listener_result["text"]], [all_text])
    evaluation_score=dict()
    evaluation_score['realism_score']=realism_score
    evaluation_score['sentiment_emb']=sentiment_emb

    evaluation_score['bleu_scores']=bleu_scores[0]

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
    OUTPUT_audio_FILE=result_img_save_path+'/output_audio'+str(1)+'.wav'
    # if os.path.exists(OUTPUT_audio_FILE):
    #     os.remove(OUTPUT_audio_FILE)
    if not os.path.exists(result_img_save_path):
        os.mkdir(result_img_save_path)
    '''edge tts'''
    # communicate = edge_tts.Communicate(all_text, VOICE)
    # # pdb.set_trace()
    # communicate.save_sync(OUTPUT_audio_FILE)
    # pdb.set_trace()

    # audio_device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # audio_model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-v1").to(audio_device)
    # tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")

    prompt = all_text#"Hey, how are you doing today?"
    # if tone_pos:
    description = "A male speaker delivers a slightly expressive and animated speech with a moderate speed and pitch. \
        The recording is of very high quality, with the speaker's voice sounding clear and very close up. The tone is positive level of "+str(tone_pos)
    # else:
    #     description = "A male speaker delivers a slightly expressive and animated speech with a moderate speed and pitch. \
    #     The recording is of very high quality, with the speaker's voice sounding clear and very close up. The tone is negative."
    
    input_ids = tokenizer(description, return_tensors="pt").input_ids.to(audio_device)
    prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(audio_device)

    generation = audio_model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
    audio_arr = generation.cpu().numpy().squeeze()

    # 获取原始speaker音频的长度以保证对齐
    asdf_aduio=np.zeros(audio_model.config.sampling_rate*3,dtype=np.float32)
    new_audio=np.concatenate((asdf_aduio,audio_arr),axis=0)

    pdb.set_trace()
    speaker_audio, sr_speaker = librosa.load(speaker_audio_path, sr=16000)
    speaker_duration = len(speaker_audio) / sr_speaker
    
    target_samples = int(speaker_duration * audio_model.config.sampling_rate)
    new_audio_arr = np.zeros(target_samples, dtype=np.float32)

    # 1. 确定speaker说话的时间段 (using whisper segments)
    speaker_active = np.zeros(target_samples, dtype=bool)
    if 'segments' in result:
        for segment in result['segments']:
            start_idx = int(segment['start'] * audio_model.config.sampling_rate)
            end_idx = int(segment['end'] * audio_model.config.sampling_rate)
            # 限制在合法的时间范围内
            start_idx = max(0, min(start_idx, target_samples - 1))
            end_idx = max(0, min(end_idx, target_samples))
            if start_idx < end_idx:
                speaker_active[start_idx:end_idx] = True

    silent_indices = np.where(~speaker_active)[0]
    if len(silent_indices) > 0:
        # Group contiguous silent regions
        breaks = np.where(np.diff(silent_indices) != 1)[0] + 1
        gap_chunks = np.split(silent_indices, breaks)
        
        # Find the longest gap to paste the audio block
        longest_gap = max(gap_chunks, key=len)
        best_gap_start = longest_gap[0]
        
        insert_len = min(len(audio_arr), target_samples - best_gap_start)
        new_audio_arr[best_gap_start : best_gap_start + insert_len] = audio_arr[:insert_len]
        
    audio_arr = new_audio_arr
    
    # 不添加额外的静音，直接保存
    new_audio = audio_arr.astype(np.float32)
    sf.write(OUTPUT_audio_FILE, new_audio, audio_model.config.sampling_rate)
    print(f'audio_generated, duration: {len(new_audio)/audio_model.config.sampling_rate:.2f}s, path: {OUTPUT_audio_FILE}')




    
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

    # video_ = ((input_latent* 0.5 + 0.5) * 255).add_(0.5).clamp_(0, 255).to(dtype=torch.uint8).cpu().permute(0, 2, 3, 1).contiguous()
    # video_save_path = os.path.join(args.save_video_path, 'sample_test_beforecrop' + str(batch_index)+'.mp4')
    # print(video_.shape)
    # imageio.mimwrite(video_save_path, video_, fps=8, quality=9)
    # print('save path {}'.format(args.save_video_path))
    # pdb.set_trace()
    input_img_list = np.split(np.uint8(np.transpose(127.5*(input_latent+1).cpu().numpy(),(0,2,3,1)))[:,:,:,::-1],input_latent.shape[0],axis=0)#[img_path, ]

    # #reverse here
    # reverse_input_img_list=input_img_list_single[::-1]#.reverse()
    # #reverse end
    # #repeat start
    # repeat_times=2
    # input_img_list=repeat_times*(input_img_list_single+reverse_input_img_list)  #
    # #repeat and concat
    # # input_img_list[sample]
    bbox_shift=0
    # pdb.set_trace()  # REMOVED DEBUG BREAKPOINT
    coord_list_single, frame_list_single = get_landmark_and_bbox(input_img_list, bbox_shift)

    input_latent_list_single = []
    print('len(frame_list_single)',len(frame_list_single))
    for bbox, frame in zip(coord_list_single, frame_list_single):
        bbox=np.array(bbox)
        bbox[bbox<0]=0
        frame=np.squeeze(frame)

        if (bbox == coord_placeholder).all():
            x1, y1, x2, y2 = [0,0,256,256]
        else:
            x1, y1, x2, y2 = bbox
            
        crop_frame = frame[y1:y2, x1:x2]
        crop_frame = cv2.resize(crop_frame,(256,256),interpolation = cv2.INTER_LANCZOS4)
        latents = vae.get_latents_for_unet(crop_frame)
        input_latent_list_single.append(latents)

    new_frame_list_single=[]
    for single_ori_img in frame_list_single:
        # keep standard orientation format
        new_frame_list_single.append( torch.permute(torch.tensor(single_ori_img[:,:,:,::-1]/127.5-1),(0,3,1,2)) )

    target_frames = len(whisper_chunks)
    
    # 1. Timeline Setup
    speaker_active_frames = np.zeros(target_frames, dtype=bool)
    if 'segments' in result:
        for segment in result['segments']:
            start_frame = int(segment['start'] * fps)
            end_frame = int(segment['end'] * fps)
            start_frame = max(0, min(start_frame, target_frames - 1))
            end_frame = max(0, min(end_frame, target_frames))
            if start_frame < end_frame:
                speaker_active_frames[start_frame:end_frame] = True

    # 2. Extract 75/25 cutoff pools
    total_gen_frames = len(input_latent_list_single)
    reply_frames_cutoff = int(total_gen_frames * 0.75)
    
    # 3. Target requirements
    target_listen = np.sum(speaker_active_frames)
    target_reply = target_frames - target_listen
    
    input_latent_list_cycle = []
    original_image_cycle = []
    coord_list_cycle = []
    
    reply_idx = 0
    listen_idx = 0
    
    for i in range(target_frames):
        if speaker_active_frames[i]:
            # Listen zone (last 25%)
            if total_gen_frames - reply_frames_cutoff > 0 and target_listen > 0:
                l_i = reply_frames_cutoff + int(listen_idx * (total_gen_frames - reply_frames_cutoff) / target_listen)
                l_i = min(l_i, total_gen_frames - 1)
            else:
                l_i = total_gen_frames - 1
            input_latent_list_cycle.append(input_latent_list_single[l_i])
            original_image_cycle.append(new_frame_list_single[l_i])
            coord_list_cycle.append(coord_list_single[l_i])
            listen_idx += 1
        else:
            # Reply zone (first 75%)
            if reply_frames_cutoff > 0 and target_reply > 0:
                r_i = int(reply_idx * reply_frames_cutoff / target_reply)
                r_i = min(r_i, reply_frames_cutoff - 1)
            else:
                r_i = 0
            input_latent_list_cycle.append(input_latent_list_single[r_i])
            original_image_cycle.append(new_frame_list_single[r_i])
            coord_list_cycle.append(coord_list_single[r_i])
            reply_idx += 1


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
    res_frame_list_tensor=[]
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
            res_frame_list_tensor.append(torch.unsqueeze(torch.tensor(res_frame.copy()),dim=0))
        for res_latent in pred_latents:
            #mask latent
            # res_latent_detrans=res_latent*0.5+0.5
            # res_latent_detrans = res_latent_detrans * (self._mask_tensor>0.5)
            res_frame_list_latent.append(res_latent)
            # cv2.imwrite(f"{result_img_save_path}/{str(i).zfill(8)}.png",res_frame)
    # return res_frame_list

    # croped_image=torch.cat(res_frame_list_tensor,dim=0).to(dtype=torch.uint8).cpu().contiguous()
    # # video_ = ((croped_image * 0.5 + 0.5) * 255).add_(0.5).clamp_(0, 255).to(dtype=torch.uint8).cpu().permute(0, 2, 3, 1).contiguous()
    # video_save_path = os.path.join(args.save_video_path, 'sample_test_afterRF_crop' + str(batch_index)+'.mp4')
    # # print(video_save_path)
    # print(croped_image.shape)
    # imageio.mimwrite(video_save_path, croped_image, fps=8, quality=9)
    # print('save path {}'.format(args.save_video_path))

    # pdb.set_trace()
    if return_image4emotion:
        interval_listener=len(res_frame_list)//num_frame_sp
        if interval_listener==0:
            out_images=res_frame_list
        else:
            out_images=res_frame_list[::interval_listener]
        return out_images[:num_frame_sp],evaluation_score
    print("pad talking image to original video")
    print('len(res_frame_list)=',len(res_frame_list))
    # pdb.set_trace()
    for i, res_frame in enumerate(tqdm(res_frame_list)):
        
        ori_frame = torch.squeeze(copy.deepcopy(original_image_cycle[i%(len(original_image_cycle))]))#.permute(2,1,0)
        # ori_frame = copy.deepcopy(res_frame_list[i%(len(res_frame_list))])
        # t_shape = ori_frame.shape#coord_list_cycle[i%(len(coord_list_cycle))]
        t_shape = coord_list_cycle[i%(len(coord_list_cycle))]
        # bbox=[0,0,t_shape[0],t_shape[1]]
        
        bbox=t_shape

        bbox=np.array(bbox)
        bbox[bbox<0]=0
        # frame=np.squeeze(frame)
        # if bbox == coord_placeholder:
        #     continue
        # x1, y1, x2, y2 = bbox

        if (bbox == coord_placeholder).all():
            # continue
            bbox=[0,0,256,256]
        # x1, y1, x2, y2 = [0,0,256,256]
    # else:
        x1, y1, x2, y2 = bbox#arr[arr < 0] = 0
        # if not args.without_landmark:
        # crop_frame = frame[y1:y2, x1:x2]




        # x1, y1, x2, y2 = bbox
        
        # try:
        res_frame = cv2.resize(res_frame.astype(np.uint8),(x2-x1,y2-y1))
        # pdb.set_trace()
        # combine_frame = get_image(ori_frame,res_frame,bbox)
        # combine_frame = get_image(np.uint8(127.5*(ori_frame.permute(1,2,0).cpu()+1))[:,:,::-1],res_frame,bbox)
        combine_frame = get_image(np.uint8(127.5*(ori_frame.permute(1,2,0).cpu()+1))[:,:,::-1],res_frame,bbox)
        #clip to -1:1,, or 0-255
        cv2.imwrite(f"{result_img_save_path}/{str(i).zfill(8)}.png",combine_frame)
        # pdb.set_trace()
        # cv2.imwrite(f"{result_img_save_path}/{str(i).zfill(8)}.png",ori_frame)
        # cv2.imwrite(f"{result_img_save_path}/{str(i).zfill(8)}.png",res_frame)
    # pdb.set_trace()
    # cmd_img2video = f"ffmpeg -y -v warning -r {fps} -f image2 -i {result_img_save_path}/%08d.png -vcodec libx264 -vf format=rgb24,scale=out_color_matrix=bt709,format=yuv420p -crf 18 temp.mp4"
    cmd_img2video = f"ffmpeg -y -v warning -r {fps} -f image2 -i {result_img_save_path}/%08d.png  -vf format=rgb24,scale=out_color_matrix=bt709,format=yuv420p -crf 18 temp2.mp4"

    print(cmd_img2video)
    os.system(cmd_img2video)
    

    # output_vid_name = os.path.join('./result_talking/siyang',"siyang.mp4")
    output_vid_name = os.path.join(args.save_video_path, 'sample_test_final' + str(batch_index)+'.mp4')
    
    # Mix both original speaker audio and generated listener audio cleanly, mapping to AAC explicitly so video players don't mute it
    cmd_combine_audio = f'ffmpeg -y -v warning -i "{OUTPUT_audio_FILE}" -i "{speaker_audio_path}" -i temp2.mp4 -filter_complex "[0:a][1:a]amix=inputs=2:duration=longest[a]" -map 2:v -map "[a]" -c:v copy -c:a aac "{output_vid_name}"'
    print(cmd_combine_audio)
    os.system(cmd_combine_audio)
    
    # os.remove("./temp.mp4")
    shutil.rmtree(result_img_save_path)
    print(f"result is save to {output_vid_name}")      

    print('the corresponding speaker path is: ', speaker_audio_path)
    # pdb.set_trace()
    # if return_image4emotion:
    #return emotion image
    # pdb.set_trace()
    interval_listener=len(res_frame_list)//num_frame_sp
    out_images=res_frame_list[::interval_listener]
    # return out_images[:num_frame_sp],evaluation_score 
    
    return out_images[:num_frame_sp],evaluation_score 


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
    clip_size=64 #64
    #
    # pdb.set_trace()
    print('check test/val matrix..')
    # if val_test == 'val':
    if 'sbg'  in platform.node() or 'dg' in platform.node() or 'xlg' in platform.node():
        args.dataset_path='/data/home/acw569/paper_face/face_react/latte/latte_reaction/val_test_matrix'
    else:
        args.dataset_path='./val_test_matrix'
    #speaker_neighbour_matrix = np.load(os.path.join(args.dataset_path, 'neighbour_emotion_val.npy'))
    # else:
    #speaker_neighbour_matrix = np.load(os.path.join(args.dataset_path, 'neighbour_emotion_test.npy'))

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
    # whisper_model = whisper.load_model("base.en")
    # Whisper放在GPU 1以节省GPU 0显存
    whisper_device = 1 if torch.cuda.device_count() > 1 else 0
    whisper_model = whisper.load_model("medium.en", device=f"cuda:{whisper_device}")
    print(f"[Memory] Loaded whisper on GPU {whisper_device}")

    # result = whisper_model.transcribe(audio_file)
    # print(result["text"])

    # sentences=result["text"].split('.')
    ##LLM processing
    # pipe = pipeline("text-generation", "meta-llama/Meta-Llama-3-8B-Instruct", torch_dtype=torch.bfloat16, device_map="auto")
    # pipe = pipeline("text-generation", "meta-llama/Meta-Llama-3-8B-Instruct", torch_dtype=torch.float16, device_map="auto")
    # LLM放在GPU 1以节省GPU 0显存
    llm_device = 1 if torch.cuda.device_count() > 1 else 0
    if 'server' in platform.node():
        pipe = pipeline("text-generation", model="/data/szl/tools/llama3", torch_dtype=torch.float16, device=llm_device)
    else:
        pipe = pipeline("text-generation", "meta-llama/Meta-Llama-3-8B-Instruct", torch_dtype=torch.float16, device=llm_device)
    print(f"[Memory] Loaded llama3 on GPU {llm_device}")

    ### deepseek
    # pipe = pipeline("text-generation", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", torch_dtype=torch.float16, device=1)
    
    #initlize chat
    
    #init talking head
    
    # load model weights
    # pdb.set_trace()
    audio_processor, vae, unet, pe = load_all_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timesteps = torch.tensor([0], device=device)

    # 显示GPU显存分配信息
    print("\n" + "="*60)
    print("[GPU Memory Status]")
    num_gpus = torch.cuda.device_count()
    for i in range(num_gpus):
        total_mem, free_mem = torch.cuda.mem_get_info(i)
        print(f"GPU {i}: Total {total_mem/1e9:.2f} GiB, Free {free_mem/1e9:.2f} GiB")
    print("="*60 + "\n")

    #end audio->talking head----------------------------------------
    using_cfg = args.cfg_scale > 1.0

    # Load model:
    latent_size = args.image_size // 8
    args.latent_size = latent_size
    model = get_models(args).to(device)
    print(f"[Memory] Latte model on GPU 0, GPU 0 current free: {torch.cuda.mem_get_info(0)[0] / 1e9:.2f} GiB")

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
    print(f"[Memory] VAE loaded on GPU 0, GPU 0 current free: {torch.cuda.mem_get_info(0)[0] / 1e9:.2f} GiB")


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
    
    # 预处理：获取speaker音频的停顿点和关键帧信息

    # 注意：pause_frames 现在将在循环内部按batch获取
    
    # Initialize score dictionary - keep only cumulative counts, not lists
    all_score = dict()
    all_score['all_realism_score'] = 0.0
    all_score['all_sentiment_score'] = 0.0
    all_score['all_valence_score'] = 0.0
    all_score['all_arousal_score'] = 0.0
    all_score['all_bleu_scores'] = 0.0
    all_score['all_rouge_scores'] = 0.0
    all_score['all_bert_scores'] = 0.0
    all_score['all_sentiment_emb'] = torch.zeros([len(dataset), 10])
    
    processed_batches = 0  # Track processed batches for memory monitoring
    # pdb.set_trace()
    # Labels to condition the model with (feel free to change):
    # double_count=0
    torch.manual_seed(0)
    for step, video_data in enumerate(loader):
        print('current step is:{}, batch_size is :{}'.format(step,batch_size))
        
        # Clean GPU and CPU memory at batch start
        torch.cuda.empty_cache()
        gc.collect()  # Force CPU garbage collection
        # if step<280:
        #     continue
        # Skip steps until we reach the resumed step
        # if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
        #     continue
        # pdb.set_trace()
        
        # 从dataloader中获取audio文件路径
        speaker_audio_path = video_data['speaker_audio_path'][0] if isinstance(video_data['speaker_audio_path'], (list, tuple)) else video_data['speaker_audio_path']
        listener_audio_path = video_data['listener_audio_path'][0] if isinstance(video_data['listener_audio_path'], (list, tuple)) else video_data['listener_audio_path']
        
        # 获取当前batch的pause_frames
        # pdb.set_trace()  # REMOVED DEBUG BREAKPOINT
        print(f"Processing speaker audio: {speaker_audio_path}")
        speaker_whisper_result = whisper_model.transcribe(speaker_audio_path)
        total_frames = video_data['img'].shape[1]
        
        # 获取speaker说话和停顿的具体时间段
        speaking_intervals, pause_intervals = get_speaker_speaking_intervals(
            speaker_whisper_result, fps=25, total_frames=total_frames
        )
        print(f"[Speaking intervals] {len(speaking_intervals)} segments: {speaking_intervals[:3]}...")
        print(f"[Pause intervals] {len(pause_intervals)} segments: {pause_intervals[:3]}...")
        
        # 获取停顿点（用于生成触发）
        pause_frames = detect_speech_pauses(speaker_whisper_result, speaker_audio_path, fps=25)
        print(f"[Pauses] Detected {len(pause_frames)} pause points: {pause_frames[:10]}...")
        
        # Clean up whisper result from memory
        del speaker_whisper_result
        gc.collect()
        
        out_samples=None
        # pdb.set_trace()
        if args.use_fp16:#args.num_frames
            z = torch.randn(batch_size, args.num_frames, 4, latent_size, latent_size, dtype=torch.float16, device=device) # b c f h w
        else:
            z = torch.randn(batch_size, args.num_frames, 4, latent_size, latent_size, device=device)#args.num_frames
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

        # pdb.set_trace()  # REMOVED DEBUG BREAKPOINT
        # 智能裁剪：根据停顿和关键帧生成反应
        total_frames = (video_data['img'].shape)[1]
        generation_points = merge_generation_triggers(
            pause_frames, 
            [],  # keyframe_indices 在实时处理中可根据需要添加
            min_interval=clip_size,
            fps=25
        )
        
        # 转换为clip索引
        clip_indices = []
        for gen_point in generation_points:
            clip_idx = gen_point['frame'] // clip_size
            if clip_idx < total_frames // clip_size:
                clip_indices.append((clip_idx, gen_point['type']))
        
        # 如果没有检测到停顿点，使用默认的均匀分布
        if not clip_indices:
            print("[Warning] No pause points detected, using uniform distribution")
            clip_indices = [(i, 'uniform') for i in range((video_data['img'].shape)[1]//clip_size)]
        
        print(f"[Generation] Processing {len(clip_indices)} clips with strategies: {[(idx, typ) for idx, typ in clip_indices[:5]]}...")
        
        # Initialize output samples concatenation list (will be processed per batch)
        all_face_images = []  # IMPORTANT: Clear this list for each batch
        out_samples = None
        last_generated_samples = None  # Keep track of last generated frames for "hold" mode
        
        for index_clip, strategy in clip_indices:
            # Determine generation mode based on speaking/pause intervals
            gen_mode, reason = get_clip_generation_mode(
                index_clip, clip_size, speaking_intervals, pause_intervals
            )
            
            # Log generation mode
            clip_frame_start = index_clip * clip_size
            print(f"[Clip {index_clip}] Mode: {gen_mode}, Reason: {reason}")
            
            # For "hold" mode during speaker speaking, use interpolation strategy
            if gen_mode == 'hold':
                # Get listener reference video
                listener_ref_clip = video_data['listener_video_clip'].to(device, non_blocking=True)[:, index_clip*clip_size:(index_clip+1)*clip_size]
                
                if last_generated_samples is not None:
                    # Use last generated samples as primary, with interpolation transition
                    print(f"  -> Using interpolated transition (75% reaction + 25% blend to reference)")
                    
                    # Decode last generated samples to pixel space for interpolation
                    b_last, f_last, c_last, h_last, w_last = last_generated_samples.shape if len(last_generated_samples.shape) == 5 else (last_generated_samples.shape[0], last_generated_samples.shape[1], last_generated_samples.shape[2], last_generated_samples.shape[3], last_generated_samples.shape[4])
                    decoded_last_batch = []
                    for batch_idx in range(b_last):
                        sample_instance = last_generated_samples[batch_idx] if len(last_generated_samples.shape) == 5 else last_generated_samples[batch_idx:batch_idx+1]
                        # Ensure correct shape for decode
                        if len(sample_instance.shape) == 4:
                            sample_instance = sample_instance.unsqueeze(0) if sample_instance.shape[0] != args.num_frames else sample_instance
                        decoded_instance = reaction_vae.decode(sample_instance / 0.18215).sample
                        decoded_last_batch.append(decoded_instance)
                    
                    generated_frames_decoded = torch.stack(decoded_last_batch, dim=0)
                    del decoded_last_batch
                    
                    # Decode reference frames for interpolation
                    if listener_ref_clip.shape[1] >= args.num_frames:
                        ref_frames = listener_ref_clip[:, :args.num_frames]
                        # Apply interpolation: 75% generated + smoothly transition to reference
                        samples_decoded = interpolate_frames_with_transition(
                            generated_frames_decoded, ref_frames, 
                            transition_ratio=0.75, clip_size=clip_size
                        )
                        print(f"    Interpolated with reference frames: 75% reaction, 25% blend")
                    else:
                        # No good reference, use decay to static
                        samples_decoded = interpolate_frames_with_transition(
                            generated_frames_decoded, None,
                            transition_ratio=0.75, clip_size=clip_size
                        )
                        print(f"    Interpolated with decay: 75% reaction, 25% fade to static")
                    
                    # Encode back to latent space
                    b_enc = samples_decoded.shape[0] if len(samples_decoded.shape) == 5 else 1
                    encoded_batch = []
                    for batch_idx in range(b_enc):
                        sample_inst = samples_decoded[batch_idx] if len(samples_decoded.shape) == 5 else samples_decoded
                        enc_inst = reaction_vae.encode(sample_inst.to(torch.float32)).latent_dist.sample() * 0.18215
                        encoded_batch.append(enc_inst)
                    samples = torch.stack(encoded_batch, dim=0) if len(samples_decoded.shape) == 5 else encoded_batch[0]
                    samples_pixels = samples_decoded  # Save for later use
                    
                elif listener_ref_clip.shape[1] == args.num_frames:
                    # Use reference frames directly
                    print(f"  -> Using reference frames for hold period (no prior generation)")
                    samples_pixels = listener_ref_clip
                    # pdb.set_trace()
                    b_enc = listener_ref_clip.shape[0] if len(listener_ref_clip.shape) == 5 else 1
                    encoded_batch = []
                    for batch_idx in range(b_enc):
                        sample_inst = listener_ref_clip[batch_idx] if len(listener_ref_clip.shape) == 5 else listener_ref_clip
                        enc_inst = reaction_vae.encode(sample_inst.to(torch.float32)).latent_dist.sample() * 0.18215
                        encoded_batch.append(enc_inst)
                    samples = torch.stack(encoded_batch, dim=0) if len(listener_ref_clip.shape) == 5 else encoded_batch[0]
                    last_generated_samples = samples
                    
                else:
                    # Use first reference frame repeated with decay interpolation
                    print(f"  -> Using first reference frame (repeated + decay)")
                    first_ref_frame = listener_ref_clip[:, 0:1]
                    samples_pixels = first_ref_frame.repeat(1, args.num_frames, 1, 1, 1)
                    
                    # Apply decay effect
                    samples_pixels = interpolate_frames_with_transition(
                        samples_pixels, None,
                        transition_ratio=0.75, clip_size=clip_size
                    )
                    
                    b_enc = samples_pixels.shape[0] if len(samples_pixels.shape) == 5 else 1
                    encoded_batch = []
                    for batch_idx in range(b_enc):
                        sample_inst = samples_pixels[batch_idx] if len(samples_pixels.shape) == 5 else samples_pixels
                        enc_inst = reaction_vae.encode(sample_inst.to(torch.float32)).latent_dist.sample() * 0.18215
                        encoded_batch.append(enc_inst)
                    samples = torch.stack(encoded_batch, dim=0) if len(samples_pixels.shape) == 5 else encoded_batch[0]
                
                # Decode the samples for final output
                b, f, c, h, w = samples.shape if len(samples.shape) == 5 else (samples.shape[0], samples.shape[1], samples.shape[2], samples.shape[3], samples.shape[4])
                decoded_batch = []
                for batch_idx in range(b):
                    sample_instance = samples[batch_idx] if len(samples.shape) == 5 else samples[batch_idx:batch_idx+1]
                    if len(sample_instance.shape) == 4:
                        sample_instance = sample_instance.unsqueeze(0) if sample_instance.shape[0] != args.num_frames else sample_instance
                    decoded_instance = reaction_vae.decode(sample_instance / 0.18215).sample
                    decoded_batch.append(decoded_instance)
                samples = torch.stack(decoded_batch, dim=0)
                del decoded_batch
                
            else:
                # Normal generation for pause periods
                x = video_data['img'].to(device, non_blocking=True)[:,index_clip*clip_size:(index_clip+1)*clip_size]#.half()
                listener_ref=video_data['listener_video_clip'].to(device, non_blocking=True)[:,index_clip*clip_size:(index_clip+1)*clip_size]#.half()
                listener_video_clip_last=video_data['listener_video_clip_last'].to(device, non_blocking=True)[:,index_clip*clip_size:(index_clip+1)*clip_size]
                sp_audio_clip=video_data['speaker_audio_clip'].to(device, non_blocking=True)
                
                if listener_ref.shape[1]<args.num_frames:
                    # pdb.set_trace()  # REMOVED DEBUG BREAKPOINT
                    print(f"Warning: listener_ref shape {listener_ref.shape[1]} < num_frames {args.num_frames}")
                
                emo_label=video_data['listener_emotion'].to(device, non_blocking=True)[:,index_clip*clip_size:(index_clip+1)*clip_size]
                speaker_emo=video_data['speaker_emotion'].to(device, non_blocking=True)[:,index_clip*clip_size:(index_clip+1)*clip_size]
                original_x=x

                #classifier free guidance, change half to zero
                if using_cfg:
                    x_uncon=torch.zeros_like(x)
                    listener_ref_uncon=torch.zeros_like(listener_ref)
                    x=torch.cat([x,x_uncon],dim=0)
                    listener_ref=torch.cat([listener_ref,listener_ref_uncon],dim=0)
                
                # Sample images using diffusion
                if args.sample_method == 'ddim':
                    samples = diffusion.ddim_sample_loop(
                        sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device,
                        original_x=x,listener_ref=listener_ref,listener_video_clip_last=listener_video_clip_last,
                        sp_audio_clip=sp_audio_clip
                    )
                elif args.sample_method == 'ddpm':
                    samples = diffusion.p_sample_loop(
                        sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device,
                        original_x=x,listener_ref=listener_ref,listener_video_clip_last=listener_video_clip_last,
                        sp_audio_clip=sp_audio_clip
                    )
                
                if args.use_fp16:
                    samples = samples.to(dtype=torch.float16)
                if using_cfg:
                    samples=samples[:len(x)//2]
                    
                b, f, c, h, w = samples.shape
                
                # VAE decode
                decoded_batch = []
                for batch_idx in range(b):
                    sample_instance = samples[batch_idx]
                    decoded_instance = reaction_vae.decode(sample_instance / 0.18215).sample
                    decoded_batch.append(decoded_instance)
                
                samples = torch.stack(decoded_batch, dim=0)
                del decoded_batch
                last_generated_samples = samples
            
            # pdb.set_trace()
            if index_clip==0:
                out_samples=samples#rearrange(samples, '(b f) c h w -> b f c h w', b=b)
            else:
                out_samples=torch.cat((out_samples,samples),dim=1)
            del samples
        # 统计生成策略
        pause_driven_count = len([1 for _, strat in clip_indices if strat == 'pause'])
        uniform_count = len([1 for _, strat in clip_indices if strat == 'uniform'])
        print(f"[Generation Summary] Total clips: {len(clip_indices)}, Pause-driven: {pause_driven_count}, Uniform: {uniform_count}")
        # #can consider do not use the decode of vae
        dim_out=out_samples.shape
        print(dim_out)
        # all_face_images=openai_deepseek_talkinghead_genall(audio_file,whisper_model,audio_processor,
        #             emo_label,out_samples.reshape(dim_out[0]*dim_out[1],dim_out[2],dim_out[3],dim_out[4]), pipe,vae, unet, pe,num_frame_sp=dim_out[0]*dim_out[1])     #openai,deepseek
        # all_face_images=torch.tensor(np.transpose(np.asarray(all_face_images),(0,3,1,2))/127.5-1,device=out_samples.device).float()
        # pdb.set_trace()
        for batch_index in range(len(out_samples)):
            
            # #save videos before RF
            # video_ = ((out_samples[batch_index] * 0.5 + 0.5) * 255).add_(0.5).clamp_(0, 255).to(dtype=torch.uint8).cpu().permute(0, 2, 3, 1).contiguous()
            # video_save_path = os.path.join(args.save_video_path, 'sample_' + str(len(out_samples)*step+batch_index)+'.mp4')
            # # print(video_save_path)
            # imageio.mimwrite(video_save_path, video_, fps=8, quality=9)
            # print('save path {}'.format(args.save_video_path))
            

            batched_faces,scores=openai_deepseek_talkinghead_genall(
            whisper_model,audio_processor,video_data['speaker_audio_path'][batch_index],video_data['listener_audio_path'][batch_index],
                    emo_label,out_samples[batch_index], pipe,vae, unet, pe,batch_index=len(out_samples)*step+batch_index,num_frame_sp=dim_out[1],num_llm_responses=len(clip_indices))     #openai,deepseek
            # calculate_talkinghead_genall(audio_file,whisper_model,audio_processor,
            #         emo_label,out_samples[batch_index], pipe,vae, unet, pe)  #llama
            all_face_images+=batched_faces
            # Accumulate only numeric values, not lists
            all_score['all_realism_score'] += float(scores['realism_score']) if isinstance(scores['realism_score'], (int, float, torch.Tensor)) else 0.0
            all_score['all_sentiment_emb'][batch_size*step+batch_index]=scores['sentiment_emb']
            # Accumulate only the sum/mean of scores, not all details
            all_score['all_bleu_scores'] += float(np.mean(scores['bleu_scores'])) if scores['bleu_scores'] else 0.0
            all_score['all_bert_scores'] += float(np.mean(scores['bert_scores'])) if scores['bert_scores'] else 0.0
            # video_ = np.transpose(np.asarray(batched_faces),(0,3,1,2))#.permute(0, 2, 3, 1).contiguous()
            
            # pdb.set_trace()
            video_ = np.asarray(np.clip(batched_faces,0,255))[:,:,:,::-1]
            video_save_path = os.path.join(args.save_video_path, 'generated_RF_' + str(len(out_samples)*step+batch_index)+'.mp4')
            print(video_save_path)
            # print(video_.shape)
            imageio.mimwrite(video_save_path, video_, fps=8, quality=9)
            
            # Explicitly delete large tensors after writing
            del video_, batched_faces, scores
            
        # Clear batch outputs to free memory
        if out_samples is not None:
            del out_samples
        # Clean up variables that may or may not have been created
        for var_name in ['x', 'listener_ref', 'listener_video_clip_last', 'sp_audio_clip', 'emo_label', 'speaker_emo']:
            if var_name in locals():
                del locals()[var_name]
        del video_data  # Free video_data from CPU memory
        
        processed_batches += 1
        print(f"[Memory] Processed {processed_batches} batches. Collector stats: {gc.get_count()}")
        
        # Force garbage collection every N batches
        if processed_batches % 5 == 0:
            gc.collect()
            print(f"[Memory] Forced garbage collection at batch {processed_batches}")
        # a=1
        # all_face_images=torch.tensor(np.transpose(np.asarray(all_face_images)[:,:,:,::-1],(0,3,1,2))/127.5-1,device=out_samples.device).float()
        # # #generate talking head
        # # result = whisper_model.transcribe(audio_file)
        # # print(result["text"])

        # # sentences=result["text"].split('.')

        # #visualize 
        

        # # video_ = ((out_samples[0] * 0.5 + 0.5) * 255).add_(0.5).clamp_(0, 255).to(dtype=torch.uint8).cpu().permute(0, 2, 3, 1).contiguous()
        # # video_save_path = os.path.join(args.save_video_path, 'sample_test_ddpm' + str(step)+'.mp4')
        # # print(video_save_path)
        # # imageio.mimwrite(video_save_path, video_, fps=8, quality=9)
        # # print('save path {}'.format(args.save_video_path))
        # # pdb.set_trace()


        # # #visualize 
        # # video_ = ((video_data['img'][0] * 0.5 + 0.5) * 255).add_(0.5).clamp_(0, 255).to(dtype=torch.uint8).cpu().permute(0, 2, 3, 1).contiguous()
        # # video_save_path = os.path.join(args.save_video_path, 'original' + str(step)+'.mp4')
        # # print(video_save_path)
        # # imageio.mimwrite(video_save_path, video_, fps=8, quality=9)
        # # print('save path {}'.format(args.save_video_path))

        # # video_ = ((out_samples[0] * 0.5 + 0.5) * 255).add_(0.5).clamp_(0, 255).to(dtype=torch.uint8).cpu().permute(0, 2, 3, 1).contiguous()
        # # video_save_path = os.path.join(args.save_video_path, 'sample_test_ddpm' + str(step)+'.mp4')
        # # print(video_save_path)
        # # imageio.mimwrite(video_save_path, video_, fps=8, quality=9)
        # # print('save path {}'.format(args.save_video_path))
        # # pdb.set_trace()

        # # pdb.set_trace()
        # #get emo feature
        # b,v,c,w,h=out_samples.shape
        # # b,v,c_l,w_l,h_l=lnd.shape

        # all_face_images=out_samples.reshape(b*v,c,w,h)
        # b,c,w,h=all_face_images.shape
        # # all_face_images=out_samples
        # id_feats,emo_fea,au_disfa,au_bp4d=encoder_emo.forward_listener(all_face_images.to(torch.device('cuda:0')))#.to(torch.device('cuda:0'))
        # # id_feats,emo_fea,au_disfa,au_bp4d=model.facial_condition.forward_listener(all_face_images,detect_face=True)
        # # id_feats,emo_fea,au_disfa,au_bp4d=model.facial_condition.forward_listener(samples)
        # # id=id.reshape(b*v,c,w,h)
        # # lnd=lnd.reshape(b*v,c_l,w_l,h_l)
        
        # # # pdb.set_trace()
        # # self.idnet.eval()
        # # id_feats = self.idnet(id)  # [B, 512]

        # #emo_feature
        # # pdb.set_trace()
        # au=torch.cat([au_bp4d, au_disfa[:, 4:5], au_disfa[:,-2:]],dim=-1)  #[8*b,15]->[b,8,15]
        # au=torch.reshape(au,[b,-1,15])

        # #
        # expression=emo_fea['expression']
        # expression_soft=Func.softmax(expression,dim=1)
        # valence=emo_fea['valence']
        # arousal=emo_fea['arousal']
        # emo=torch.cat([expression_soft,torch.unsqueeze(valence,dim=-1),torch.unsqueeze(arousal,dim=-1)],dim=-1)
        # emo=torch.reshape(emo,[b,-1,10])
        # current_emo=emo#torch.cat([emo,au],dim=-1)
        # # current_emo=torch.cat([emo,au],dim=-1)

        # # pdb.set_trace()
        # # #get emo label from listener image
        # # if all_face_images.shape[0]!=listener_ref.shape[0]:
        # #     listener_emotion_pred_list.append(current_emo.cpu())
        # #     listener_emotion_gt_list.append(emo_label.cpu())
        # #     speaker_emotion_gt_list.append(speaker_emo)
        # #     continue
        # listener_ref_4emo=listener_ref.reshape(all_face_images.shape[0],all_face_images.shape[1],all_face_images.shape[2],all_face_images.shape[3])
        # id_feats,emo_fea,au_disfa,au_bp4d=encoder_emo.forward_listener(listener_ref_4emo.to(torch.device('cuda:1')))#.to(torch.device('cuda:0'))
        
        # au=torch.cat([au_bp4d, au_disfa[:, 4:5], au_disfa[:,-2:]],dim=-1)  #[8*b,15]->[b,8,15]
        # au=torch.reshape(au,[b,-1,15])

        # #
        # expression=emo_fea['expression']
        # expression_soft=Func.softmax(expression,dim=1)
        # valence=emo_fea['valence']
        # arousal=emo_fea['arousal']
        # emo=torch.cat([expression_soft,torch.unsqueeze(valence,dim=-1),torch.unsqueeze(arousal,dim=-1)],dim=-1)
        # emo=torch.reshape(emo,[b,-1,10])
        # listener_gt_frommodel_emo=emo#torch.cat([emo,au],dim=-1)
        # # listener_gt_frommodel_emo=torch.cat([emo,au],dim=-1)

        # #
        # # pdb.set_trace()
        # listener_emotion_pred_list.append(current_emo.cpu())
        # listener_emotion_gt_list.append(listener_gt_frommodel_emo.cpu())
        # # listener_emotion_gt_list.append(emo_label.cpu())
        # speaker_emotion_gt_list.append(speaker_emo)
        

    pdb.set_trace()
    listener_emotion_pred = torch.cat(listener_emotion_pred_list, dim = 0)
    listener_emotion_gt = torch.cat(listener_emotion_gt_list, dim = 0)

    speaker_emotion_gt=torch.cat(speaker_emotion_gt_list, dim = 0)

    all_listener_emotion_pred_list.append(listener_emotion_pred.unsqueeze(1))
    all_listener_emotion_pred = torch.cat(all_listener_emotion_pred_list, dim=1)
    # if not os.path.exists(args.save_video_path):
    #     os.makedirs(args.save_video_path)


    #save and load parameter
    torch.save(all_listener_emotion_pred, "./emo_pred_24v4.pt")
    torch.save(speaker_emotion_gt, "./speaker_emo_gt_24v4.pt")
    torch.save(listener_emotion_gt, "./listener_emo_gt_24v4.pt")
    torch.save(all_score['all_sentiment_emb'][:len(listener_emotion_gt)], "./llm_emo_pred_24v4.pt")
    torch.save(all_score,"./llm_all_score_24v4.pt")

    # emo_pred=torch.load("./emo_pred.pt")
    # speaker_emo_gt=torch.load("./speaker_emo_gt.pt")
    # listener_emo_gt=torch.load("./listener_emo_gt.pt")
    # llm_emo_pred=torch.load('./llm_emo_pred.pt')
    # emo_pred=torch.load("./emo_pred.pt")
    # If you have problems running function compute_TLCC_mp, please replace this function with function compute_TLCC
    # TLCC = compute_TLCC(all_listener_emotion_pred.cpu(), speaker_emotion_gt.cpu())
    # pdb.set_trace()
    # If you have problems running function compute_FRC_mp, please replace this function with function compute_FRC

    #for react 2025
    # torch.save(speaker_emotions_input_all,'./speaker_emo_gt_19.gt')
    # torch.save(listener_emotion_preds_lists_all,'./emo_pred_19.gt')
    # torch.save(listener_emotion_GTs_all,'./listener_emo_gt_19.gt')
    print("-----------------Evaluating Metric-----------------")
# clip_size
    listener_emo_gt_reshape=listener_emo_gt.reshape(len(llm_emo_pred),clip_size,1,10)
    language_mse=torch.nn.functional.mse_loss(all_score['all_sentiment_emb'][:len(listener_emotion_gt)],torch.mean(listener_emo_gt_reshape[:,:,:,:10],1))#check listener_emotion_gt.shape
    print('language_mse',language_mse)
    # pdb.set_trace()
    realism_score=all_score['all_realism_score']/len(listener_emotion_gt)
    bleu_scores=all_score['all_bleu_scores']/len(listener_emotion_gt)
    bert_scores=all_score['all_bert_scores']/len(listener_emotion_gt)
    print('realism_score',realism_score)
    print('bleu_scores',bleu_scores)
    print('bert_scores',bert_scores)
    
    # #if 2025
    # results = compute_metrics(
    #         speaker_emotions_input_all,
    #         listener_emotion_preds_lists_all,
    #         listener_emotion_GTs_all,
    #     )

    #elif 2024
    if len(all_listener_emotion_pred)<len(dataset):
        repeat_times=len(dataset)//len(all_listener_emotion_pred)
        fake_all_listener_emotion_pred=all_listener_emotion_pred.repeat(repeat_times,1,1,1)
        fake_listener_emotion_gt=listener_emotion_gt.repeat(repeat_times,1,1)
        FRC = compute_FRC(args, fake_all_listener_emotion_pred.cpu(), fake_listener_emotion_gt.cpu(), val_test='test')

        # If you have problems running function compute_FRD_mp, please replace this function with function compute_FRD
        FRD = compute_FRD(args, fake_all_listener_emotion_pred.cpu(), fake_listener_emotion_gt.cpu(), val_test='test')
    else:
        FRC = compute_FRC(args, all_listener_emotion_pred.cpu(), listener_emotion_gt.cpu(), val_test='test')

        # If you have problems running function compute_FRD_mp, please replace this function with function compute_FRD
        FRD = compute_FRD(args, all_listener_emotion_pred.cpu(), listener_emotion_gt.cpu(), val_test='test')

    FRDvs = compute_FRDvs(all_listener_emotion_pred.cpu())
    FRVar  = compute_FRVar(all_listener_emotion_pred.cpu())
    smse  = compute_s_mse(all_listener_emotion_pred.cpu())
    # print('TLCC',TLCC)
    print("TLCC: %.4f; FRC: %.4f;FRD: %.4f;\n FRDvs: %.4f;FRVar: %.4f;smse: %.4f;" % (TLCC, FRC, FRD, FRDvs, FRVar, smse))

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/reaction/reaction_test.yaml")
    
    if 'sbg'  in platform.node() or 'dg' in platform.node() or 'xlg' in platform.node():
        parser.add_argument("--ckpt", type=str, default="/gpfs/scratch/acw569/checkpoint/latte_check_new/021-Latte-B-4-F64S3-reaction-Amp/checkpoints/0060000.pt")
    else:
        parser.add_argument("--ckpt", type=str, default="/data/szl/code/latte_reaction_new/latte_reaction_check/023-Latte-B-4-F64S3-reaction-Amp/checkpoints/0260000.pt")#2024 f64?
        # parser.add_argument("--ckpt", type=str, default="/data/szl/code/latte_reaction_new/latte_reaction_check/023-Latte-B-4-F64S3-reaction-Amp/checkpoints/0260000.pt")#2024 f64?

    #2025 v5 /data/scratch/acw569/checkpoint/latte_check_new/055-Latte-S-4-F32S3-reaction-Amp/checkpoints/0120000.pt 
    #2024 v5 /data/scratch/acw569/checkpoint/latte_check_new/054-Latte-S-4-F32S3-reaction-Amp/checkpoints/0170000.pt
    #2025 v4 /data/scratch/acw569/checkpoint/latte_check_new/043-Latte-S-4-F32S3-reaction-Amp/checkpoints/0190000.pt
    #2024 v4 /data/scratch/acw569/checkpoint/latte_check_new/015-Latte-S-4-F32S3-reaction-Amp/checkpoints/0260000.pt bad generation
    # /data/scratch/acw569/checkpoint/reaction/previous/0240000.pt
    # /data/scratch/acw569/checkpoint/latte_check/018-Latte-S-4-F32S3-reaction-Amp/checkpoints/0180000.pt
    # parser.add_argument("--ckpt", type=str, default="/import/nobackup_mmv_ioannisp/zs003/code/face_reaction/latte_reaction/results/123-Latte-S-2-F32S3-reaction-Amp/checkpoints/0210000.pt")
    #/import/nobackup_mmv_ioannisp/zs003/code/face_reaction/latte_reaction/checkpoint_fromhpc/0250000.pt
    
    #wrong label:/import/nobackup_mmv_ioannisp/zs003/code/face_reaction/latte_reaction/results/092-Latte-S-2-F32S3-reaction-Amp/checkpoints/0240000.pt
    parser.add_argument("--save_video_path", type=str, default="./sample_videos/")
    args = parser.parse_args()
    omega_conf = OmegaConf.load(args.config)
    omega_conf.ckpt = args.ckpt
    omega_conf.save_video_path = args.save_video_path
    main(omega_conf)
