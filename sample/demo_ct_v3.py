# All rights reserved.

'''this version suffer from:
1.face inconsistency as I used different synthesized facial videos
2.video inconsistency because the parltts could not generate the same audio tone(speaker).
'''
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
# from transformers import AutoTokenizer
import torchaudio
from transformers import AutoFeatureExtractor, AutoTokenizer, set_seed
import soundfile as sf
from decord import VideoReader
from decord import cpu as decord_cpu
from PIL import Image
from torchvision import transforms as T
import json as _json


# ─── Helper: load frames from a video file path ─────────────────────────────
def load_frames_from_video_path(video_path, start_frame, num_frames, img_size=256):
    """Read *num_frames* consecutive frames starting at *start_frame* from an
    mp4 file.  Returns ``(clip, total_length)`` where *clip* is a float32
    tensor of shape ``[1, num_frames, 3, img_size, img_size]`` in ``[-1, 1]``
    (same normalisation as the dataset ``Transform`` class) and *total_length*
    is the total number of frames in the video.
    """
    with open(video_path, 'rb') as f:
        vr = VideoReader(f, ctx=decord_cpu(0))
    total_length = len(vr)
    _tf = T.Compose([T.Resize(img_size), T.ToTensor(),
                     T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    clip = []
    for i in range(start_frame, start_frame + num_frames):
        idx = min(i, total_length - 1)
        frame = vr[idx]
        img = Image.fromarray(frame.asnumpy())
        clip.append(_tf(img).unsqueeze(0))
    clip = torch.cat(clip, dim=0).unsqueeze(0)  # [1, num_frames, 3, H, W]
    return clip, total_length


# ─── Helper: LLM selects which speaker segments need a listener response ─────
def llm_select_response_segments(pipe, full_transcript, segments):
    """Send the entire speaker transcript (with per-segment timestamps) to the
    LLM and ask it to decide at which segments the listener should produce a
    verbal response.  Returns a filtered list of segment dicts (a subset of
    *segments*).
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
    # Parse JSON list from LLM output
    try:
        indices = _json.loads(raw)
        if not isinstance(indices, list):
            indices = list(range(len(segments)))
    except (_json.JSONDecodeError, ValueError):
        nums = re.findall(r'\d+', raw)
        indices = [int(n) for n in nums if int(n) < len(segments)]
        if not indices:
            indices = list(range(len(segments)))

    # ── Post-processing: fix noisy LLM output ──────────────────────────────
    # 1. Deduplicate and clamp to valid range
    seen = set()
    unique_indices = []
    for idx in indices:
        if isinstance(idx, (int, float)):
            idx = int(idx)
        else:
            continue
        if 0 <= idx < len(segments) and idx not in seen:
            seen.add(idx)
            unique_indices.append(idx)
    indices = unique_indices
    print(f"[LLM Timing] Raw LLM output: {raw!r}")
    print(f"[LLM Timing] Deduplicated indices: {indices}")

    # 2. Compute the gap (silence) after each selected segment and prefer the
    #    segment with the largest gap — that's the most natural response window.
    #    Keep at least one segment; if multiple segments share the max gap, keep
    #    all of them.
    if len(indices) > 1:
        gap_info = []   # list of (gap_seconds, idx)
        for idx in indices:
            seg_end = segments[idx].get('end', 0.0)
            # Find the start of the next segment in the *full* segment list
            if idx + 1 < len(segments):
                next_start = segments[idx + 1].get('start', seg_end)
            else:
                # Last segment: gap extends to "infinity" — use a large value
                next_start = seg_end + 999.0
            gap = max(0.0, next_start - seg_end)
            gap_info.append((gap, idx))
        max_gap = max(g for g, _ in gap_info)
        # Keep all segments whose gap is within 80 % of the max gap (tolerance
        # for cases where several gaps are similarly large)
        gap_threshold = max_gap * 0.8
        indices = [idx for gap, idx in gap_info if gap >= gap_threshold]
        print(f"[LLM Timing] Gaps: {[(f'{g:.2f}s', i) for g, i in gap_info]}  "
              f"→ kept indices with gap ≥ {gap_threshold:.2f}s: {indices}")

    selected = [segments[i] for i in indices if 0 <= i < len(segments)]
    if not selected:
        selected = segments
    print(f"[LLM Timing] Final selected segment indices: {indices}")
    return selected


# ─── Helper: create side-by-side speaker (left) + listener (right) video ─────
def create_side_by_side_video(speaker_video_path, speaker_audio_path,
                              listener_frames, listener_audio_path,
                              output_path, fps=25):
    """Create a video with the speaker on the left and the generated listener
    on the right, with mixed audio from both the speaker and listener tracks.

    *listener_frames*: list of numpy uint8 HWC-RGB arrays (the final pipeline
    output).
    *listener_audio_path*: path to the TTS-generated listener wav file.
    """
    # ── Read speaker frames ──
    with open(speaker_video_path, 'rb') as f:
        vr = VideoReader(f, ctx=decord_cpu(0))
    n_speaker = len(vr)
    n_listener = len(listener_frames)
    n_out = max(n_speaker, n_listener)

    target_h = 256

    combined = []
    for i in range(n_out):
        # Speaker frame
        sp_idx = min(i, n_speaker - 1)
        sp_frame = vr[sp_idx].asnumpy()  # HWC-RGB
        sp_frame = cv2.resize(sp_frame, (target_h, target_h))  # square for simplicity

        # Listener frame
        li_idx = min(i, n_listener - 1)
        li_frame = np.array(listener_frames[li_idx], dtype=np.uint8)
        li_frame = cv2.resize(li_frame, (target_h, target_h))

        side = np.concatenate([sp_frame, li_frame], axis=1)  # [H, 2W, 3]
        combined.append(side)

    combined = np.stack(combined, axis=0)  # [N, H, 2W, 3]

    # ── Write temporary video (no audio) ──
    tmp_vid = output_path.replace('.mp4', '_tmp_sbs.mp4')
    imageio.mimwrite(tmp_vid, combined, fps=fps, quality=9)

    # ── Mix audio ──
    sp_audio, sp_sr = sf.read(speaker_audio_path)
    if sp_audio.ndim == 2:
        sp_audio = sp_audio.mean(axis=1)
    li_audio, li_sr = sf.read(listener_audio_path)
    if li_audio.ndim == 2:
        li_audio = li_audio.mean(axis=1)
    # Resample listener audio to speaker sample-rate if needed
    if li_sr != sp_sr:
        x_old = np.linspace(0, 1, len(li_audio))
        x_new = np.linspace(0, 1, int(len(li_audio) * sp_sr / li_sr))
        li_audio = np.interp(x_new, x_old, li_audio).astype(np.float32)
    # Pad shorter track
    max_len = max(len(sp_audio), len(li_audio))
    if len(sp_audio) < max_len:
        sp_audio = np.pad(sp_audio, (0, max_len - len(sp_audio)))
    if len(li_audio) < max_len:
        li_audio = np.pad(li_audio, (0, max_len - len(li_audio)))
    mixed = (sp_audio.astype(np.float32) + li_audio.astype(np.float32))
    peak = np.max(np.abs(mixed)) + 1e-8
    if peak > 1.0:
        mixed = mixed / peak
    mixed_path = output_path.replace('.mp4', '_mixed.wav')
    sf.write(mixed_path, mixed, sp_sr)

    # ── Combine video + mixed audio via ffmpeg ──
    cmd = f"ffmpeg -y -v warning -i {tmp_vid} -i {mixed_path} -c:v copy -c:a aac -shortest {output_path}"
    print(f"[Side-by-side] {cmd}")
    os.system(cmd)
    # Clean up temp files
    if os.path.exists(tmp_vid):
        os.remove(tmp_vid)
    if os.path.exists(mixed_path):
        os.remove(mixed_path)
    print(f"[Side-by-side] Saved to {output_path}")


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
audio_device = "cuda:1" if torch.cuda.device_count() > 1 else ("cuda:0" if torch.cuda.is_available() else "cpu")

audio_model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-v1").to(audio_device)
tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")

# ─── Timeline-aware temporal frame stretching ────────────────────────────────

def _fill_frames(result, src_frames, start_idx, end_idx):
    """Linearly interpolate src_frames into result[start_idx:end_idx]."""
    if end_idx <= start_idx or len(src_frames) == 0:
        return
    n_target = end_idx - start_idx
    n_src = len(src_frames)
    for j in range(n_target):
        if start_idx + j >= len(result):
            break
        if n_target == 1 or n_src == 1:
            result[start_idx + j] = np.array(src_frames[0], dtype=np.uint8)
            continue
        src_pos = j * (n_src - 1) / (n_target - 1)
        lo = int(src_pos)
        hi = min(lo + 1, n_src - 1)
        alpha = src_pos - lo
        if alpha < 1e-6:
            result[start_idx + j] = np.array(src_frames[lo], dtype=np.uint8)
        else:
            blended = (1 - alpha) * np.array(src_frames[lo], dtype=np.float32) + \
                       alpha       * np.array(src_frames[hi], dtype=np.float32)
            result[start_idx + j] = np.clip(blended, 0, 255).astype(np.uint8)


def stretch_frames_to_timeline(frames, segment_list, target_total_frames,
                                video_fps=25, frames_per_seg=64):
    """
    Stretch N_segs×64 generated frames to exactly target_total_frames,
    aligning listener timing with the speaker's timeline:

    For each segment k:
      • passive (first 25% = 16 frames)  → speaker talking window
                                            [segment[k].start … segment[k].end]
      • active  (last  75% = 48 frames)  → silence gap
                                            [segment[k].end … segment[k+1].start]
                                            (or video end for the last segment)

    Any uncovered leading frames (before segment 0 starts) are filled with the
    first available frame, keeping the output fully contiguous.
    """
    if not frames or target_total_frames <= 0:
        return frames

    n_segs = len(segment_list)
    passive_per_seg = int(0.25 * frames_per_seg)   # 16
    active_per_seg  = frames_per_seg - passive_per_seg  # 48

    result = [None] * target_total_frames

    for seg_i, seg in enumerate(segment_list):
        seg_start_frame = max(0, int(seg.get('start', 0) * video_fps))
        seg_end_frame   = min(target_total_frames, int(seg.get('end',   0) * video_fps))

        # silence gap: from end of this segment to start of next
        if seg_i + 1 < n_segs:
            next_start_frame = min(target_total_frames,
                                   int(segment_list[seg_i + 1].get('start', seg_end_frame) * video_fps))
        else:
            next_start_frame = target_total_frames

        # Source frames for this segment
        src_passive = frames[seg_i * frames_per_seg : seg_i * frames_per_seg + passive_per_seg]
        src_active  = frames[seg_i * frames_per_seg + passive_per_seg : (seg_i + 1) * frames_per_seg]

        # Passive frames → speaker talking window
        if seg_end_frame > seg_start_frame and src_passive:
            _fill_frames(result, src_passive, seg_start_frame, seg_end_frame)

        # Active frames → silence gap (listener's turn to speak)
        if next_start_frame > seg_end_frame and src_active:
            _fill_frames(result, src_active, seg_end_frame, next_start_frame)

    # Forward-fill any None slots (e.g., gap before first segment)
    fallback = None
    for i in range(len(result)):
        if result[i] is not None:
            fallback = result[i]
        elif fallback is not None:
            result[i] = fallback.copy()

    # Backward-fill leading Nones (before first segment)
    fallback = None
    for i in range(len(result) - 1, -1, -1):
        if result[i] is not None:
            fallback = result[i]
        elif fallback is not None:
            result[i] = fallback.copy()

    # Final safety: replace any remaining None with zeros
    empty = np.zeros_like(frames[0]) if frames else np.zeros((256, 256, 3), dtype=np.uint8)
    result = [f if f is not None else empty for f in result]

    return result

# ─────────────────────────────────────────────────────────────────────────────

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

def openai_deepseek_talkinghead_genall(audio_file,listener_audio_file,whisper_model,audio_processor,
    speaker_audio_path,listener_audio_path, emo_label,input_latent, pipe,vae, unet, pe,batch_index=0,batch_size=8,fps=10,
        result_img_save_path='./temp',VOICE = "en-GB-SoniaNeural",return_image4emotion=False,num_frame_sp=32,
        segment_list=None):
    """
    input_latent: Tensor [N_segs*64, C, H, W] - all dialog segments stacked.
    segment_list: list of dicts with keys 'text', 'start', 'end' - one per 64-frame block.
    """
    # pdb.set_trace()
    # for step,clip_generated in enumerate(input_latent):

    frames_per_seg = 64  # each segment window is always 64 frames

    # ---- Derive segment texts from segment_list or fall back to Whisper transcription ----
    if segment_list is not None and len(segment_list) > 0:
        seg_texts = [seg.get("text", "").strip() for seg in segment_list]
        n_segments = len(seg_texts)
        print(f"[Timeline] Processing {n_segments} speaker segments on timeline")
    else:
        result = whisper_model.transcribe(speaker_audio_path)
        print(result["text"])
        seg_texts = [result["text"].strip()]
        n_segments = 1

    # ---- Reshape input_latent into [n_segments, 64, C, H, W] ----
    if input_latent.dim() == 4:
        total_frames_il = input_latent.shape[0]
        if total_frames_il == frames_per_seg * n_segments:
            seg_latents = input_latent.reshape(n_segments, frames_per_seg, *input_latent.shape[1:])
        elif total_frames_il == frames_per_seg:
            seg_latents = input_latent.unsqueeze(0)  # single segment [1,64,C,H,W]
        else:
            # fallback: use first 64 frames only
            seg_latents = input_latent[:frames_per_seg].unsqueeze(0)
            n_segments = 1
            seg_texts = seg_texts[:1]
    else:
        seg_latents = input_latent[:1]
        n_segments = 1
        seg_texts = seg_texts[:1]

    # ---- Emotion features from first-segment latent ----
    id_feats, emo_fea, au_disfa, au_bp4d = encoder_emo.forward_listener(seg_latents[0].to(torch.device('cuda:0')))
    expression = emo_fea['expression'].to(torch.device('cuda:0'))
    expression_soft = F.softmax(expression, dim=1)
    valence = emo_fea['valence'].to(torch.device('cuda:0'))
    arousal = emo_fea['arousal'].to(torch.device('cuda:0'))

    # ---- Listener reference transcription (once, for scoring) ----
    listener_result = whisper_model.transcribe(listener_audio_path)
    listener_reference_sentences = listener_result["text"].split('.')

    # ---- Rolling conversational LLM dialogue (one turn per timeline segment) ----
    openai_api_key = "None"
    openai_api_base = "http://0.0.0.0:8000/v1"
    client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)

    random_number = np.random.randint(low=0, high=100)
    tone_pos = 1
    all_text = ''
    current_text = ''

    # TTS speaking rate (words per second) used to cap reply length to fit the listener's reaction window
    _TTS_WORDS_PER_SEC = 2.5

    chat = [
        {"role": "system", "content": (
            "We are having a real-time conversation. Pretend you are a person with a specific personality "
            "(a sales representative), subject to the random seed: " + str(random_number) +
            ". For each speaker turn, generate a natural response that fits the live dialogue flow. "
            "Each reply must not exceed the word-count limit stated in the user message."
        )}
    ]

    seg_reply_texts = []
    is_last_seg = False
    for seg_i, speaker_text in enumerate(seg_texts):
        is_last_seg = (seg_i == len(seg_texts) - 1)
        # Compute per-segment word limit from the GAP after the segment
        # (the active window where the listener actually speaks), not the
        # segment's own duration.
        if segment_list is not None and seg_i < len(segment_list):
            _seg_info = segment_list[seg_i]
            _seg_end = float(_seg_info.get('end', 0.64))
            if seg_i + 1 < len(segment_list):
                _next_start = float(segment_list[seg_i + 1].get('start', _seg_end))
                _gap_dur = max(0.5, _next_start - _seg_end)
            else:
                # Last segment: no next-segment constraint, use a large gap so
                # the word count is not artificially capped at 50.
                _gap_dur = None
        else:
            _gap_dur = frames_per_seg / 25.0  # fallback

        if _gap_dur is None or is_last_seg:
            # Last segment: no upper word-count cap
            _max_words = None
        else:
            _max_words = max(5, round(_gap_dur * _TTS_WORDS_PER_SEC))

        # Build user message; from turn 2 onwards, carry forward emotion context
        _word_limit_hint = f" [Reply in at most {_max_words} words.]" if _max_words is not None else ""
        if seg_i == 0:
            base_text = speaker_text if speaker_text else "Hello."
            user_msg = f"{base_text}{_word_limit_hint}"
        else:
            # Append previous assistant turn to maintain dialogue continuity
            chat.append({"role": "assistant", "content": seg_reply_texts[-1]})
            emo_hint = (
                " (Please match the emotional tone — "
                "Neutral={:.2f}, Happy={:.2f}, Sad={:.2f}, valence={:.2f}, arousal={:.2f}. "
                "Do not include these scores in your reply.)"
            ).format(
                float(expression_soft[:, 0].mean()),
                float(expression_soft[:, 1].mean()),
                float(expression_soft[:, 2].mean()),
                float(valence.mean()),
                float(arousal.mean())
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

    # Close conversation with last assistant turn (needed for scoring)
    chat.append({"role": "assistant", "content": seg_reply_texts[-1] if seg_reply_texts else ''})
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
    OUTPUT_audio_FILE = result_img_save_path + '/output_audio' + str(1) + '.wav'
    if not os.path.exists(result_img_save_path):
        os.mkdir(result_img_save_path)

    sr = audio_model.config.sampling_rate
    video_fps_audio = 25

    description_base = ("Lea's voice is monotone, with a very close recording that almost has no background noise. "
                        "The tone is positive level of " + str(tone_pos))

    _seg_list: list = segment_list if segment_list else [{'start': 0.0, 'end': 0.64, 'text': ''}]
    # Use the actual speaker audio duration so the output matches the speaker timeline exactly
    _sp_audio_info = sf.info(speaker_audio_path)
    total_duration_sec = _sp_audio_info.duration
    total_samples = int(sr * total_duration_sec)

    # ── Build output audio: silence while speaker talks, TTS in the gaps ─────
    # Layout for each segment i:
    #   [seg.start … seg.end]         → silence  (speaker is talking)
    #   [seg.end   … next_seg.start]  → listener TTS reply
    # Any leading silence [0 … first_seg.start] is already zeros.
    # ─────────────────────────────────────────────────────────────────────────
    new_audio = np.zeros(total_samples, dtype=np.float32)

    # ── Pass 1: collect all active (non-empty) segments with their window info ──
    active_segs = []
    for seg_i, seg in enumerate(_seg_list):
        seg_end_sec   = seg.get('end', 0.64)
        reply_text    = seg_reply_texts[seg_i] if seg_i < len(seg_reply_texts) else ''

        if seg_i + 1 < len(_seg_list):
            next_start_sec = _seg_list[seg_i + 1].get('start', seg_end_sec)
        else:
            next_start_sec = total_duration_sec

        active_start_s = int(sr * seg_end_sec)
        active_end_s   = min(total_samples, int(sr * next_start_sec))
        target_samples = active_end_s - active_start_s
        seg_dur        = seg_end_sec - seg.get('start', 0.0)
        active_dur     = max(0.0, next_start_sec - seg_end_sec)

        if not reply_text.strip() or target_samples <= 0:
            pdb.set_trace()
            print(f'[Audio seg {seg_i}] silence={seg_dur:.2f}s  gap={active_dur:.2f}s  (no reply text)')
            continue

        active_segs.append({
            'seg_i':          seg_i,
            'reply_text':     reply_text,
            'active_start_s': active_start_s,
            'active_end_s':   active_end_s,
            'target_samples': target_samples,
            'seg_dur':        seg_dur,
            'active_dur':     active_dur,
        })

    # ── Pass 2: concatenate all reply texts and generate TTS in one call ──────
    if active_segs:
        concat_text = " ".join(s['reply_text'] for s in active_segs)
        print(f'[Audio] Generating TTS for {len(active_segs)} segments in one call. text={repr(concat_text[:120])}')

        input_ids_all  = tokenizer(description_base, return_tensors="pt").input_ids.to(audio_device)
        prompt_ids_all = tokenizer(concat_text,      return_tensors="pt").input_ids.to(audio_device)
        gen_all        = audio_model.generate(input_ids=input_ids_all, prompt_input_ids=prompt_ids_all)
        tts_all        = gen_all.cpu().numpy().squeeze().astype(np.float32)
        total_tts_samples = len(tts_all)

        # ── Pass 3: split tts_all by character-length proportion of each text ──
        char_lengths  = [len(s['reply_text']) for s in active_segs]
        total_chars   = max(sum(char_lengths), 1)
        cumulative    = 0
        tts_ptr       = 0

        for idx, seg_info in enumerate(active_segs):
            cumulative += char_lengths[idx]
            if idx < len(active_segs) - 1:
                tts_end = int(total_tts_samples * cumulative / total_chars)
            else:
                tts_end = total_tts_samples  # give remainder to last segment

            tts_seg      = tts_all[tts_ptr:tts_end].copy()
            orig_seg_len = len(tts_seg)
            tts_ptr      = tts_end

            target_samples = seg_info['target_samples']
            active_start_s = seg_info['active_start_s']
            active_end_s   = seg_info['active_end_s']

            # Resample this slice to exactly fill the active gap window
            if len(tts_seg) != target_samples and target_samples > 0:
                x_old  = np.linspace(0, 1, max(len(tts_seg), 1))
                x_new  = np.linspace(0, 1, target_samples)
                tts_seg = np.interp(x_new, x_old, tts_seg).astype(np.float32)
            tts_seg = tts_seg[:target_samples]  # safety clamp

            new_audio[active_start_s:active_end_s] = tts_seg
            speed_ratio = orig_seg_len / max(1, target_samples)
            print(f'[Audio seg {seg_info["seg_i"]}] silence={seg_info["seg_dur"]:.2f}s  '
                  f'TTS_gap={seg_info["active_dur"]:.2f}s  speed={speed_ratio:.2f}x  '
                  f'reply={repr(seg_info["reply_text"][:60])}')

    sf.write(OUTPUT_audio_FILE, new_audio, sr)
    print('audio_generated, path is: ', OUTPUT_audio_FILE)
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
    
    total_frames = len(input_latent)
    num_silent_frames = 0  # all frames processed through talking head
    num_talk_frames = total_frames
    pdb.set_trace()

    # Pad or trim whisper_chunks to match total_frames
    # if len(whisper_chunks) < total_frames:
    #     pad_len = total_frames - len(whisper_chunks)
    #     if len(whisper_chunks) > 0:
    #         whisper_chunks = whisper_chunks + [whisper_chunks[-1]] * pad_len
    #     else:
    #         whisper_chunks = whisper_chunks * total_frames  # fallback: repeat
    # elif len(whisper_chunks) > total_frames:
    #     whisper_chunks = whisper_chunks[:total_frames]

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
    input_latent_talk = input_latent  # all frames go through talking head
    input_img_list = np.split(np.uint8(np.transpose(127.5*(input_latent_talk+1).cpu().numpy(),(0,2,3,1)))[:,:,:,::-1],input_latent_talk.shape[0],axis=0)#[img_path, ]

    # #reverse here
    # reverse_input_img_list=input_img_list_single[::-1]#.reverse()
    # #reverse end
    # #repeat start
    # repeat_times=2
    # input_img_list=repeat_times*(input_img_list_single+reverse_input_img_list)  #
    # #repeat and concat
    # # input_img_list[sample]
    bbox_shift=0
    coord_list_single, frame_list_single = get_landmark_and_bbox(input_img_list, bbox_shift)


    #reverse and repeat here
    repeat_times=2
    coord_list_reverse=coord_list_single[::-1]
    frame_list_reverse=frame_list_single[::-1]
    coord_list=repeat_times*(coord_list_single+coord_list_reverse)
    frame_list=repeat_times*(frame_list_single+frame_list_reverse)
    # with open(crop_coord_save_path, 'wb') as f:
    #     pickle.dump(coord_list, f)
    input_latent_list = []
    croped_image=[]
    # pdb.set_trace()
    # print('len(coord_list)',coord_list)
    print('len(frame_list)',len(frame_list))
    for bbox, frame in zip(coord_list, frame_list):
        bbox=np.array(bbox)
        bbox[bbox<0]=0
        frame=np.squeeze(frame)
        # if bbox == coord_placeholder:
        #     continue
        # x1, y1, x2, y2 = bbox

        if (bbox == coord_placeholder).all():
            # continue
            x1, y1, x2, y2 = [0,0,256,256]
        else:
            x1, y1, x2, y2 = bbox#arr[arr < 0] = 0
        # if not args.without_landmark:
        crop_frame = frame[y1:y2, x1:x2]
        # else:
        #     crop_frame = frame
        # print('frame.shape:' ,frame.shape)
        # print('crop_frame.shape:' ,crop_frame.shape)
        # print('x1, y1, x2, y2:' ,x1, y1, x2, y2)
        crop_frame = cv2.resize(crop_frame,(256,256),interpolation = cv2.INTER_LANCZOS4)
        latents = vae.get_latents_for_unet(crop_frame)
        croped_image.append(torch.unsqueeze(torch.tensor(crop_frame),dim=0))
        input_latent_list.append(latents)
    # print(crop_frame.shape)
    # croped_image=torch.cat(croped_image,dim=0).to(dtype=torch.uint8).cpu().contiguous()
    # # video_ = ((croped_image * 0.5 + 0.5) * 255).add_(0.5).clamp_(0, 255).to(dtype=torch.uint8).cpu().permute(0, 2, 3, 1).contiguous()
    # video_save_path = os.path.join(args.save_video_path, 'sample_test_aftercrop' + str(batch_index)+'.mp4')
    # # print(video_save_path)
    # print(croped_image.shape)
    # imageio.mimwrite(video_save_path, croped_image, fps=8, quality=9)
    # print('save path {}'.format(args.save_video_path))
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
    input_latent_list_cycle = input_latent_list #+ input_latent_list[::-1]
    
    # original_image_cycle= np.split(input_latent,input_latent.shape[0],axis=0)#+ np.split(input_latent,input_latent.shape[0],axis=0)[::-1]
    # pdb.set_trace()
    new_frame_list=[]
    for single_ori_img in frame_list:
        new_frame_list.append( torch.permute(torch.tensor(single_ori_img[:,:,:,::-1]/127.5-1),(0,3,1,2))   )
    original_image_cycle=new_frame_list
    # original_image_cycle= np.split(frame_list,frame_list.shape[0],axis=0)#+ np.split(input_latent,input_latent.shape[0],axis=0)[::-1]

    
    laten_repeat=max(len(whisper_chunks)//len(input_latent_list_cycle),1)
    # pdb.set_trace()
    vae_encode_latents_new=[]
    for single_latent in original_image_cycle:
        repeated_latent=laten_repeat * [single_latent]
        vae_encode_latents_new+=repeated_latent
    original_image_cycle=vae_encode_latents_new
    
    # frame_list_cycle = frame_list + frame_list[::-1]
    # coord_list_cycle = coord_list + coord_list[::-1]
    coord_list_cycle = coord_list #+ coord_list[::-1]

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

    croped_image=torch.cat(res_frame_list_tensor,dim=0).to(dtype=torch.uint8).cpu().contiguous()
    # video_ = ((croped_image * 0.5 + 0.5) * 255).add_(0.5).clamp_(0, 255).to(dtype=torch.uint8).cpu().permute(0, 2, 3, 1).contiguous()
    video_save_path = os.path.join(args.save_video_path, 'sample_test_afterRF_crop' + str(batch_index)+'.mp4')
    # print(video_save_path)
    print(croped_image.shape)
    imageio.mimwrite(video_save_path, croped_image, fps=8, quality=9)
    print('save path {}'.format(args.save_video_path))

    pdb.set_trace()
    if return_image4emotion:
        interval_listener=len(res_frame_list)//num_frame_sp
        if interval_listener==0:
            out_images=res_frame_list
        else:
            out_images=res_frame_list[::interval_listener]
        return out_images[:num_frame_sp], evaluation_score, OUTPUT_audio_FILE
    print("pad talking image to original video")
    
    final_output_frames = []

    for i, res_frame in enumerate(tqdm(res_frame_list)):
        save_idx = i
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
        cv2.imwrite(f"{result_img_save_path}/{str(save_idx).zfill(8)}.png",combine_frame)
        final_output_frames.append(combine_frame[:,:,::-1])
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
    
    cmd_combine_audio = f"ffmpeg -y -v warning -i {OUTPUT_audio_FILE} -i temp2.mp4 {output_vid_name}"
    print(cmd_combine_audio)
    os.system(cmd_combine_audio)
    
    # os.remove("./temp.mp4")
    # Clean up intermediate PNG frames
    for _png in sorted(os.listdir(result_img_save_path)):
        if _png.endswith('.png'):
            os.remove(os.path.join(result_img_save_path, _png))
    print(f"result is save to {output_vid_name}")      

    print('the corresponding speaker path is: ', speaker_audio_path)
    
    out_images = final_output_frames
    
    return out_images[:num_frame_sp], evaluation_score, OUTPUT_audio_FILE


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
    USE_REAL_LISTENER = True  # True: use real listener video, skip diffusion; False: run diffusion as before
    audio_file='/import/nobackup_mmv_ioannisp/zs003/face_react/new_audio_folder/all_together/NoXI/026_2016-04-06_Nottingham/Expert_video/3.wav'
    listener_audio_file='/import/nobackup_mmv_ioannisp/zs003/face_react/new_audio_folder/all_together/NoXI/026_2016-04-06_Nottingham/Novice_video/3.wav'
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
    # whisper_model = whisper.load_model("medium.en")
    whisper_model = whisper.load_model("large-v3")
    # result = whisper_model.transcribe(audio_file)
    # print(result["text"])

    # sentences=result["text"].split('.')
    ##LLM processing
    # pipe = pipeline("text-generation", "meta-llama/Meta-Llama-3-8B-Instruct", torch_dtype=torch.bfloat16, device_map="auto")
    # pipe = pipeline("text-generation", "meta-llama/Meta-Llama-3-8B-Instruct", torch_dtype=torch.float16, device_map="auto")
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
    torch.manual_seed(0)
    T_len = 0
    for step, video_data in enumerate(loader):
        
        
        for batch_index in range(batch_size):
            # ── Derive T_len from speaker video path instead of video_data['img'] ──
            

            # if 'session4/Camera-2024-10-23-134159-134313' not in _first_sp_path:
            #     continue
            # pdb.set_trace()
            print('current step is:{}, batch_size is :{}'.format(step,batch_size))
            # speaker_audio_path = video_data['speaker_audio_path'][batch_index]
            # listener_audio_path = video_data['listener_audio_path'][batch_index]
            # speaker_video_path_cur = video_data['speaker_video_path'][batch_index]
            # listener_video_path_cur = video_data['listener_video_path'][batch_index]
            speaker_audio_path = '/data/szl/dataset/react2025/audio/speaker/session4/Camera-2024-06-26-093649-093642.wav'#video_data['speaker_audio_path'][batch_index]
            listener_audio_path = '/data/szl/dataset/react2025/audio/listener/session4/Camera-2024-06-26-093649-093642.wav'#video_data['listener_audio_path'][batch_index]
            speaker_video_path_cur = '/data/szl/dataset/react2025/video-face-crop/speaker/session4/Camera-2024-06-26-093649-093642.mp4'#video_data['speaker_video_path'][batch_index]
            listener_video_path_cur = '/data/szl/dataset/react2025/video-face-crop/listener/session4/Camera-2024-06-26-093649-093642.mp4'#video_data['listener_video_path'][batch_index]
            # pdb.set_trace()
            _first_sp_path = speaker_video_path_cur#video_data['speaker_video_path'][batch_index]
            with open(_first_sp_path, 'rb') as _f:
                _vr_tmp = VideoReader(_f, ctx=decord_cpu(0))
            T_len = len(_vr_tmp)
            del _vr_tmp
            print(speaker_video_path_cur,speaker_audio_path)
            result = whisper_model.transcribe(speaker_audio_path)
            segments = result.get("segments", [])
            # segments = segments[:4]
            if len(segments) == 0:
                segments = [{"start": 0.0, "end": 16/25.0, "text": ""}]

            # ── Feature 3: LLM selects which segments need a listener response ──
            full_transcript = " ".join([seg.get("text", "") for seg in segments])
            all_segments = list(segments)  # full Whisper segment list before LLM filtering
            segments = llm_select_response_segments(pipe, full_transcript, all_segments)

            # Update each selected segment's text to span [prev_boundary : cur_boundary]
            # so the LLM reply covers ALL speaker content since the last response.
            # E.g. if selected indices are [2, 5, 8] out of 10 total segments:
            #   response 0 uses text from all_segments[0:3]   (segments 0–2)
            #   response 1 uses text from all_segments[3:6]   (segments 3–5)
            #   response 2 uses text from all_segments[6:9]   (segments 6–8)
            #   response 3 uses text from all_segments[9:10]  (segments 9–end)
            selected_with_cumulative = []
            prev_orig_idx = -1
            last_orig_idx = -1
            for sel_seg in segments:
                cur_orig_idx = next(
                    (i for i, s in enumerate(all_segments) if s is sel_seg), None
                )
                if cur_orig_idx is None:
                    selected_with_cumulative.append(sel_seg)
                    continue
                span_text = " ".join(
                    s.get("text", "").strip()
                    for s in all_segments[prev_orig_idx + 1 : cur_orig_idx + 1]
                )
                new_seg = dict(sel_seg)
                new_seg["text"] = span_text
                selected_with_cumulative.append(new_seg)
                prev_orig_idx = cur_orig_idx
                last_orig_idx = cur_orig_idx

            # Append a tail response for [last_selected+1 : end] if any segments remain
            if last_orig_idx + 1 < len(all_segments):
                tail_text = " ".join(
                    s.get("text", "").strip()
                    for s in all_segments[last_orig_idx + 1 :]
                )
                tail_seg = dict(all_segments[-1])
                tail_seg["text"] = tail_text
                selected_with_cumulative.append(tail_seg)

            segments = selected_with_cumulative
            pdb.set_trace()
            all_face_images = []
            seg_samples_list = []  # collect [64, C, H, W] per segment
                
            for seg_idx, segment in enumerate(segments):
                end_time = segment["end"]
                end_frame = int(end_time * 25)
                start_frame = max(0, end_frame - 16)
                if start_frame >= T_len:
                    start_frame = max(0, T_len - clip_size)

                if USE_REAL_LISTENER:
                    # ── Feature 1: use real listener video, skip diffusion ──
                    print(f"[Seg {seg_idx}] Using REAL listener video (diffusion skipped)")
                    listener_clip, _ = load_frames_from_video_path(
                        listener_video_path_cur, start_frame, clip_size)
                    seg_out = listener_clip.squeeze(0).to(device)  # [64, 3, H, W]

                    # Still need emo_label for downstream evaluation
                    emo_label = video_data['listener_emotion'][batch_index:batch_index+1, start_frame:start_frame+clip_size].to(device, non_blocking=True)
                    act_len_emo = emo_label.shape[1]
                    if act_len_emo < clip_size:
                        pad_len_emo = clip_size - act_len_emo
                        if act_len_emo > 0:
                            emo_label = torch.cat([emo_label, emo_label[:, -1:].repeat(1, pad_len_emo, 1)], dim=1)
                else:
                    # ── Original diffusion generation code (preserved intact) ──
                    torch.manual_seed(0)
                    if args.use_fp16:
                        z = torch.randn(1, args.num_frames, 4, latent_size, latent_size, dtype=torch.float16, device=device)
                    else:
                        z = torch.randn(1, args.num_frames, 4, latent_size, latent_size, device=device)
                        
                    if using_cfg:
                        z = torch.cat([z, z], 0)
                        model_kwargs = dict(y=None, cfg_scale=args.cfg_scale, use_fp16=args.use_fp16)
                        sample_fn = model.forward_with_cfg_reaction
                    else:
                        sample_fn = model.forward
                        model_kwargs = dict(y=None, use_fp16=args.use_fp16)

                    # Feature 4: read speaker frames from speaker_video_path
                    x, _ = load_frames_from_video_path(speaker_video_path_cur, start_frame, clip_size)
                    x = x.to(device, non_blocking=True)
                    listener_ref = video_data['listener_video_clip'][batch_index:batch_index+1, start_frame:start_frame+clip_size].to(device, non_blocking=True)
                    listener_video_clip_last = video_data['listener_video_clip_last'][batch_index:batch_index+1, start_frame:start_frame+clip_size].to(device, non_blocking=True)
                    sp_audio_clip = video_data['speaker_audio_clip'][batch_index:batch_index+1, start_frame:start_frame+clip_size].to(device, non_blocking=True)
                    emo_label = video_data['listener_emotion'][batch_index:batch_index+1, start_frame:start_frame+clip_size].to(device, non_blocking=True)
                    
                    act_len = x.shape[1]
                    if act_len < clip_size:
                        pad_len = clip_size - act_len
                        if act_len > 0:
                            x = torch.cat([x, x[:, -1:].repeat(1, pad_len, 1, 1, 1)], dim=1)
                            listener_ref = torch.cat([listener_ref, listener_ref[:, -1:].repeat(1, pad_len, 1, 1, 1)], dim=1)
                            listener_video_clip_last = torch.cat([listener_video_clip_last, listener_video_clip_last[:, -1:].repeat(1, pad_len, 1, 1, 1)], dim=1)
                            sp_audio_clip = torch.cat([sp_audio_clip, sp_audio_clip[:, -1:].repeat(1, pad_len, 1)], dim=1)
                            emo_label = torch.cat([emo_label, emo_label[:, -1:].repeat(1, pad_len, 1)], dim=1)
                        else:
                            print(f"Warning: Segment {seg_idx} has 0 actual length! Skipping this segment.")
                            continue

                    original_x = x

                    if using_cfg:
                        x_uncon = torch.zeros_like(x)
                        listener_ref_uncon = torch.zeros_like(listener_ref)
                        x = torch.cat([x, x_uncon], dim=0)
                        listener_ref = torch.cat([listener_ref, listener_ref_uncon], dim=0)

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
                        samples = samples[:len(x)//2]
                    
                    b, f, c, h, w = samples.shape
                    decoded_batch = []
                    for b_idx in range(b):
                        sample_instance = samples[b_idx]
                        decoded_instance = reaction_vae.decode(sample_instance / 0.18215).sample
                        decoded_batch.append(decoded_instance)
                    samples = torch.stack(decoded_batch, dim=0)
                    seg_out = samples[0]  # [64, C, H, W] for this segment

                seg_samples_list.append(seg_out)

            # Stack all segment tensors: [N_segs*64, C, H, W]
            if len(seg_samples_list) == 0:
                print(f"No valid segments for batch_index={batch_index}, skipping.")
                continue
            stacked_out = torch.cat(seg_samples_list, dim=0)  # [N_segs*64, C, H, W]
            n_valid_segs = len(seg_samples_list)
            valid_segments = segments[:n_valid_segs]

            # Call talkinghead function ONCE with all N segments stacked + their metadata
            batched_faces, scores, listener_audio_file_path = openai_deepseek_talkinghead_genall(
                audio_file, listener_audio_file, whisper_model, audio_processor,
                speaker_audio_path, listener_audio_path,
                emo_label, stacked_out, pipe, vae, unet, pe,
                batch_index=(batch_size * step + batch_index),
                num_frame_sp=stacked_out.shape[0],
                segment_list=valid_segments
            )
            all_face_images += batched_faces
            
            all_score['all_realism_score'] += scores['realism_score']
            all_score['all_sentiment_emb'][batch_size*step+batch_index] = scores['sentiment_emb']
            all_score['all_bleu_scores'] += scores['bleu_scores']
            all_score['all_bert_scores'] += scores['bert_scores']
            
            # ── Stretch generated frames to match speaker's total frame count ──
            print(f"Stretching {len(all_face_images)} generated frames → {T_len} speaker frames")
            all_face_images = stretch_frames_to_timeline(
                all_face_images,
                valid_segments,
                target_total_frames=T_len,
                video_fps=25,
                frames_per_seg=64
            )
            print(f"After stretch: {len(all_face_images)} frames")
            
            video_ = np.asarray(np.clip(all_face_images, 0, 255))[:, :, :, ::-1]
            video_save_path = os.path.join(args.save_video_path, 'generated_RF_dialog_' + str(batch_size*step+batch_index)+'.mp4')
            print(video_save_path)
            imageio.mimwrite(video_save_path, video_, fps=25, quality=9)

            # ── Feature 2: side-by-side speaker (left) + listener (right) video ──
            sbs_path = os.path.join(args.save_video_path, 'side_by_side_' + str(batch_size*step+batch_index) + '.mp4')
            create_side_by_side_video(
                speaker_video_path_cur, speaker_audio_path,
                all_face_images, listener_audio_file_path,
                sbs_path, fps=25
            )
            
        # skip over loop variable legacy accumulation log since logic was completely refactored
            
        # pdb.set_trace()
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
