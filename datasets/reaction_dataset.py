import os
import torch
from torch.utils import data
from torchvision import transforms
import copy
import numpy as np
import pickle
from tqdm import tqdm
import random,math
import time
import pandas as pd
from PIL import Image
import soundfile as sf
import cv2
from torch.utils.data import DataLoader
from multiprocessing import Pool
import torchaudio
from scipy.io import loadmat
torchaudio.set_audio_backend("sox_io")
from functools import cmp_to_key
import pdb
import av
from decord import VideoReader
from decord import cpu, gpu
from torch import distributed
import platform
from pathlib import Path
import hydra
class Transform(object):
    def __init__(self, img_size=256, crop_size=224):#align with unet 256 resolution
        self.img_size = img_size
        self.crop_size = crop_size

    def __call__(self, img):
        # normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
        #                                  std=[0.5, 0.5, 0.5])
        transform = transforms.Compose([
            # transforms.Resize(self.img_size),
            # transforms.CenterCrop(self.crop_size),
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5])
        ])
        img = transform(img)
        return img


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')
def pil_loader_noconvert(path):
    img = Image.open(path)
    return img

def extract_video_features(video_path, img_transform):
    video_list = []
    video = cv2.VideoCapture(video_path)
    n_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        frame = img_transform(Image.fromarray(frame[:, :, ::-1])).unsqueeze(0)
        video_list.append(frame)
    video_clip = torch.cat(video_list, axis=0)
    return video_clip, fps, n_frames


def extract_audio_features(audio_path, fps, n_frames):
    # video_id = osp.basename(audio_path)[:-4]
    audio, sr = sf.read(audio_path)
    if audio.ndim == 2:
        audio = audio.mean(-1)
    frame_n_samples = int(sr / fps)
    curr_length = len(audio)
    target_length = frame_n_samples * n_frames
    if curr_length > target_length:
        audio = audio[:target_length]
    elif curr_length < target_length:
        audio = np.pad(audio, [0, target_length - curr_length])
    shifted_n_samples = 0
    curr_feats = []
    for i in range(n_frames):
        curr_samples = audio[i*frame_n_samples:shifted_n_samples + i*frame_n_samples + frame_n_samples]
        curr_mfcc = torchaudio.compliance.kaldi.mfcc(torch.from_numpy(curr_samples).float().view(1, -1), sample_frequency=sr, use_energy=True)
        curr_mfcc = curr_mfcc.transpose(0, 1) # (freq, time)
        curr_mfcc_d = torchaudio.functional.compute_deltas(curr_mfcc)
        curr_mfcc_dd = torchaudio.functional.compute_deltas(curr_mfcc_d)
        curr_mfccs = np.stack((curr_mfcc.numpy(), curr_mfcc_d.numpy(), curr_mfcc_dd.numpy())).reshape(-1)
        curr_feat = curr_mfccs
        # rms = librosa.feature.rms(curr_samples, sr).reshape(-1)
        # zcr = librosa.feature.zero_crossing_rate(curr_samples, sr).reshape(-1)
        # curr_feat = np.concatenate((curr_mfccs, rms, zcr))

        curr_feats.append(curr_feat)

    curr_feats = np.stack(curr_feats, axis=0)
    return curr_feats


class ReactionDataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self,root_path, real_dataset_path, split, img_size=256, crop_size=224, clip_length=751, fps=25,
                 load_audio=True, load_video_s=True, load_video_l=True, load_emotion_s=False, load_emotion_l=False,
                 load_3dmm_s=False, load_3dmm_l=False, load_ref=True,
                 repeat_mirrored=True,interval=10):
        """
        Args:
            root_path: (str) Path to the data folder.
            split: (str) 'train' or 'val' or 'test' split.
            img_size: (int) Size of the image.
            crop_size: (int) Size of the crop.
            clip_length: (int) Number of frames in a clip.
            fps: (int) Frame rate of the video.
            load_audio: (bool) Whether to load audio features.
            load_video_s: (bool) Whether to load speaker video features.
            load_video_l: (bool) Whether to load listener video features.
            load_emotion: (bool) Whether to load emotion labels.
            load_3dmm: (bool) Whether to load 3DMM parameters.
            repeat_mirrored: (bool) Whether to extend dataset with mirrored speaker/listener. This is used for val/test.
        """

        self._root_path = root_path
        self._img_loader = pil_loader
        self._clip_length = clip_length
        self._fps = fps
        self._split = split
        self.interval=interval

        self._data_path = os.path.join(self._root_path, self._split)
        self.real_dataset_path=os.path.join(real_dataset_path, self._split)
        self._list_path = pd.read_csv(os.path.join(self._root_path, self._split + '.csv'), header=None, delimiter=',')
        self._list_path = self._list_path.drop(0)
        self._new_list_path = pd.read_csv(os.path.join(self._root_path, self._split + '_new.csv'), header=None, delimiter=',')
        self._new_list_path = self._new_list_path.drop(0)

        self.load_audio = load_audio
        self.load_video_s = load_video_s
        self.load_video_l = load_video_l
        self.load_3dmm_s = load_3dmm_s
        self.load_3dmm_l = load_3dmm_l
        self.load_emotion_s = load_emotion_s
        self.load_emotion_l = load_emotion_l
        self.load_ref = load_ref

        # self.real_dataset_path=real_dataset_path
        #
        # self._audio_path = os.path.join(self._data_path, 'Audio_files')
        # self._video_path = os.path.join(self._data_path, 'Video_files')
        # self._emotion_path = os.path.join(self._data_path, 'Emotion')
        # self._3dmm_path = os.path.join(self._data_path, '3D_FV_files')
        self._audio_path = os.path.join(self.real_dataset_path, 'Audio_files')
        self._video_path = os.path.join(self.real_dataset_path, 'Video_files')
        self._emotion_path = os.path.join(self.real_dataset_path, 'Emotion')
        self._3dmm_path = os.path.join(self.real_dataset_path, '3D_FV_files')

        self.mean_face = torch.FloatTensor(
            np.load('./external/FaceVerse/mean_face.npy').astype(np.float32)).view(1, 1, -1)
        self.std_face = torch.FloatTensor(
            np.load('./external/FaceVerse/std_face.npy').astype(np.float32)).view(1, 1, -1)

        self._transform = Transform(img_size, crop_size)
        self._transform_3dmm = transforms.Lambda(lambda e: (e - self.mean_face))

        speaker_path = list(self._list_path.values[:, 1])
        listener_path = list(self._list_path.values[:, 2])
        new_speaker_path = list(self._new_list_path.values[:, 1])
        new_listener_path = list(self._new_list_path.values[:, 2])
        # pdb.set_trace()
        if self._split in ["val","test"] or repeat_mirrored:  # training is always mirrored as data augmentation
            speaker_path_tmp = speaker_path + listener_path
            listener_path_tmp = listener_path + speaker_path
            speaker_path = speaker_path_tmp
            listener_path = listener_path_tmp

            new_speaker_path_tmp = new_speaker_path + new_listener_path
            new_listener_path_tmp = new_listener_path + new_speaker_path
            new_speaker_path = new_speaker_path_tmp
            new_listener_path = new_listener_path_tmp

        self.data_list = []
        for i, (sp, lp,ns,nl) in enumerate(zip(speaker_path, listener_path,new_speaker_path,new_listener_path)):
            if 'UDIVA' in sp:
                continue
            if 'RECOLA' in sp:
                continue
            ab_speaker_video_path = os.path.join(self._video_path, sp)
            ab_speaker_audio_path = os.path.join(self._audio_path, sp + '.wav')
            #ab_speaker_emotion_path = os.path.join(self._emotion_path, sp + '.csv')
            ab_speaker_emotion_path = os.path.join(self._emotion_path, ns + '.csv')
            ab_speaker_3dmm_path = os.path.join(self._3dmm_path, sp + '.npy')

            ab_listener_video_path = os.path.join(self._video_path, lp)
            ab_listener_audio_path = os.path.join(self._audio_path, lp + '.wav')
            #ab_listener_emotion_path = os.path.join(self._emotion_path, lp + '.csv')
            ab_listener_emotion_path = os.path.join(self._emotion_path, nl + '.csv')
            ab_listener_3dmm_path = os.path.join(self._3dmm_path, lp + '.npy')

            if not all(map(os.path.exists, [ab_speaker_video_path+"/1.png", ab_speaker_audio_path, ab_speaker_emotion_path, ab_speaker_3dmm_path, ab_listener_video_path+"/1.png", ab_listener_audio_path, ab_listener_emotion_path, ab_listener_3dmm_path])):
                pdb.set_trace()
                continue
            self.data_list.append(
            {'speaker_video_path': ab_speaker_video_path, 'speaker_audio_path': ab_speaker_audio_path,
                'speaker_emotion_path': ab_speaker_emotion_path, 'speaker_3dmm_path': ab_speaker_3dmm_path,
                'listener_video_path': ab_listener_video_path, 'listener_audio_path': ab_listener_audio_path,
                'listener_emotion_path': ab_listener_emotion_path, 'listener_3dmm_path': ab_listener_3dmm_path})

        self._len = len(self.data_list)
        print(len(self.data_list))

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        # seq_len, fea_dim
        data = self.data_list[index]

        # ========================= Data Augmentation ==========================
        changed_sign = 0
        # if self._split == 'train':  # only done at training time
        #     changed_sign = random.randint(0, 1)

        speaker_prefix = 'speaker' if changed_sign == 0 else 'listener'
        listener_prefix = 'listener' if changed_sign == 0 else 'speaker'

        # ========================= Load Speaker & Listener video clip ==========================
        speaker_video_path = data[f'{speaker_prefix}_video_path']
        listener_video_path = data[f'{listener_prefix}_video_path']

        #if self.load_video_s or self.load_video_l or self.load_ref:  # otherwise, no need to compute these image paths
        img_paths = os.listdir(speaker_video_path)
        total_length = len(img_paths)
        img_paths = sorted(img_paths, key=cmp_to_key(lambda a, b: int(a[:-4]) - int(b[:-4])))
        # cp = random.randint(0, total_length - 1 - self._clip_length) if self._clip_length < total_length else 0
        cp = random.randint(0, total_length - 1 - self._clip_length*self.interval) if (self._clip_length*self.interval) < total_length else 0
        # cp=0
        # img_paths = img_paths[cp: cp + self._clip_length]
        img_paths = img_paths[cp:cp + self._clip_length*self.interval:self.interval]

        speaker_video_clip = 0
        if self.load_video_s:
            clip = []
            for img_path in img_paths:
                img = self._img_loader(os.path.join(speaker_video_path, img_path))
                img = self._transform(img)
                clip.append(img.unsqueeze(0))
            speaker_video_clip = torch.cat(clip, dim=0)

        # listener video clip only needed for val/test
        listener_video_clip = 0
        if self.load_video_l:
            clip = []
            for img_path in img_paths:
                img = self._img_loader(os.path.join(listener_video_path, img_path))
                img = self._transform(img)
                clip.append(img.unsqueeze(0))
            listener_video_clip = torch.cat(clip, dim=0)


        # ========================= Load Speaker audio clip (listener audio is NEVER needed) ==========================
        listener_audio_clip, speaker_audio_clip = 0, 0
        if self.load_audio:
            speaker_audio_path = data[f'{speaker_prefix}_audio_path']
            speaker_audio_clip = extract_audio_features(speaker_audio_path, self._fps, total_length)
            speaker_audio_clip = speaker_audio_clip[cp:cp + self._clip_length*self.interval:self.interval]


        # ========================= Load Speaker & Listener emotion ==========================
        listener_emotion, speaker_emotion = 0, 0
        if self.load_emotion_l:
            listener_emotion_path = data[f'{listener_prefix}_emotion_path']
            listener_emotion = pd.read_csv(listener_emotion_path, header=None, delimiter=',')
            listener_emotion = torch.from_numpy(np.array(listener_emotion.drop(0)).astype(np.float32))[
                               cp:cp + self._clip_length*self.interval:self.interval]

        if self.load_emotion_s:
            speaker_emotion_path = data[f'{speaker_prefix}_emotion_path']
            speaker_emotion = pd.read_csv(speaker_emotion_path, header=None, delimiter=',')
            speaker_emotion = torch.from_numpy(np.array(speaker_emotion.drop(0)).astype(np.float32))[
                              cp:cp + self._clip_length*self.interval:self.interval]

        # ========================= Load Listener 3DMM ==========================
        listener_3dmm = 0
        if self.load_3dmm_l:
            listener_3dmm_path = data[f'{listener_prefix}_3dmm_path']
            listener_3dmm = torch.FloatTensor(np.load(listener_3dmm_path)).squeeze()
            listener_3dmm = listener_3dmm[cp:cp + self._clip_length*self.interval]
            listener_3dmm = self._transform_3dmm(listener_3dmm)[0]

        speaker_3dmm = 0
        if self.load_3dmm_s:
            speaker_3dmm_path = data[f'{speaker_prefix}_3dmm_path']
            speaker_3dmm = torch.FloatTensor(np.load(speaker_3dmm_path)).squeeze()
            speaker_3dmm = speaker_3dmm[cp:cp + self._clip_length*self.interval:self.interval]
            speaker_3dmm = self._transform_3dmm(speaker_3dmm)[0]

        # ========================= Load Listener Reference ==========================
        listener_reference = 0
        if self.load_ref:
            img_paths = os.listdir(listener_video_path)
            img_paths = sorted(img_paths, key=cmp_to_key(lambda a, b: int(a[:-4]) - int(b[:-4])))
            listener_reference = self._img_loader(os.path.join(listener_video_path, img_paths[0]))
            listener_reference = self._transform(listener_reference)

        # return speaker_video_clip, speaker_audio_clip, speaker_emotion, speaker_3dmm, listener_video_clip, listener_audio_clip, listener_emotion, listener_3dmm, listener_reference
        return {'img': speaker_video_clip, 'index': index,'speaker_audio_clip': speaker_audio_clip,'speaker_emotion': speaker_emotion,
                                'speaker_3dmm': speaker_3dmm,'listener_video_clip': listener_video_clip,'listener_emotion': listener_emotion,
                                'listener_3dmm': listener_3dmm,'listener_reference': listener_reference,'speaker_emotion_path':speaker_emotion_path,'cp':[cp,cp + self._clip_length*self.interval]}
    def __len__(self):
        return self._len
class ReactionDataset_readvideo(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self,root_path, real_dataset_path, split, img_size=256, crop_size=224, clip_length=4, fps=25,
                 load_audio=True, load_video_s=True, load_video_l=True, load_emotion_s=False, load_emotion_l=False,
                 load_3dmm_s=False, load_3dmm_l=False, load_ref=True,
                 repeat_mirrored=True,interval=5):
        """
        Args:
            root_path: (str) Path to the data folder.
            split: (str) 'train' or 'val' or 'test' split.
            img_size: (int) Size of the image.
            crop_size: (int) Size of the crop.
            clip_length: (int) Number of frames in a clip.
            fps: (int) Frame rate of the video.
            load_audio: (bool) Whether to load audio features.
            load_video_s: (bool) Whether to load speaker video features.
            load_video_l: (bool) Whether to load listener video features.
            load_emotion: (bool) Whether to load emotion labels.
            load_3dmm: (bool) Whether to load 3DMM parameters.
            repeat_mirrored: (bool) Whether to extend dataset with mirrored speaker/listener. This is used for val/test.
        """

        self._root_path = root_path
        self._img_loader = pil_loader
        self._clip_length = clip_length
        self.interval=interval
        self._fps = fps
        self._split = split

        self._data_path = os.path.join(self._root_path, self._split)
        if 'sbg'  in platform.node() or 'rdg' in platform.node():
            self.real_dataset_path=real_dataset_path
        else:
            self.real_dataset_path=os.path.join(real_dataset_path, self._split)
        self._list_path = pd.read_csv(os.path.join(self._root_path, self._split + '.csv'), header=None, delimiter=',')
        
        self._new_list_path = self._list_path.copy()#pd.read_csv(os.path.join(self._root_path, self._split + '_new.csv'), header=None, delimiter=',')
        self._list_path = self._list_path.drop(0)
        self._new_list_path = self._new_list_path.drop(0)

        self.load_audio = load_audio
        self.load_video_s = load_video_s
        self.load_video_l = load_video_l
        self.load_3dmm_s = load_3dmm_s
        self.load_3dmm_l = load_3dmm_l
        self.load_emotion_s = load_emotion_s
        self.load_emotion_l = load_emotion_l
        self.load_ref = load_ref

        # self.real_dataset_path=real_dataset_path
        #
        # self._audio_path = os.path.join(self._data_path, 'Audio_files')
        # self._video_path = os.path.join(self._data_path, 'Video_files')
        # self._emotion_path = os.path.join(self._data_path, 'Emotion')
        # self._3dmm_path = os.path.join(self._data_path, '3D_FV_files')
        # self.reaction_filepath24='/import/nobackup_mmv_ioannisp/zs003/react_2024'#include emotion and data matrix
        if 'sbg'  in platform.node() or 'dg' in platform.node():
            # self._audio_path = os.path.join('/data/scratch/acw569/face_react/from_hengde/', self._split+'/Audio_files')#'/data/scratch/acw569/face_react/audios'#os.path.join(self.real_dataset_path, 'Audio_files')
            # # self._video_path = os.path.join('/data/scratch/acw569/face_react/from_hengde/', self._split+'/Video_files')#os.path.join(self.real_dataset_path, 'aligned_faces')
            # self._video_path = '/data/scratch/acw569/face_react/new_crop'
            # self._emotion_path = os.path.join('/data/scratch/acw569/face_react/from_hengde/', self._split+'/Emotion')#'/data/scratch/acw569/face_react/New_Emotion'
            # self._3dmm_path = os.path.join('/data/scratch/acw569/face_react/from_hengde/', self._split+'/3D_FV_files')#'/data/scratch/acw569/face_react/3dmm_param_all/3DMM_param'
            self._audio_path = os.path.join('/data/scratch/acw569/face_react/new_reaction_2024data', self._split+'/Audio_files')#'/data/scratch/acw569/face_react/audios'#os.path.join(self.real_dataset_path, 'Audio_files')
            # self._video_path = os.path.join('/data/scratch/acw569/face_react/from_hengde/', self._split+'/Video_files')#os.path.join(self.real_dataset_path, 'aligned_faces')
            self._video_path = '/data/scratch/acw569/face_react/new_reaction_2024data/cropped_face'
            self._emotion_path = os.path.join('/data/scratch/acw569/face_react/new_reaction_2024data', self._split+'/Emotion')#'/data/scratch/acw569/face_react/New_Emotion'
            self._3dmm_path = os.path.join('/data/scratch/acw569/face_react/new_reaction_2024data', self._split+'/3D_FV_files')#'/data/scratch/acw569/face_react/3dmm_param_all/3DMM_param'
        elif 'server' in platform.node():
            self._audio_path = os.path.join('/data/szl/dataset/data', self._split+'/Audio_files')#'/data/scratch/acw569/face_react/audios'#os.path.join(self.real_dataset_path, 'Audio_files')
            # self._video_path = os.path.join('/data/scratch/acw569/face_react/from_hengde/', self._split+'/Video_files')#os.path.join(self.real_dataset_path, 'aligned_faces')
            self._video_path = '/data/szl/dataset/data/cropped_face'
            self._emotion_path = os.path.join('/data/szl/dataset/data', self._split+'/Emotion')#'/data/scratch/acw569/face_react/New_Emotion'
            self._3dmm_path = os.path.join('/data/szl/dataset/data', self._split+'/3D_FV_files')#'/data/scratch/acw569/face_react/3dmm_param_all/3DMM_param'
         
        else:
            self.emo_filepath24='/import/nobackup_mmv_ioannisp/zs003/react_2024'#include emotion and data matrix
        
            self._audio_path = '/import/nobackup_mmv_ioannisp/zs003/face_react/new_audio_folder/all_together'#os.path.join(self.real_dataset_path, 'Audio_files')
            self._video_path = os.path.join(self.real_dataset_path, 'Video_files')
            self._emotion_path = '/import/nobackup_mmv_ioannisp/zs003/face_react/face_v2_newlabel/train/New_Emotion'
            # pdb.set_trace()
            self._emotion_path = os.path.join(self.emo_filepath24, self._split+'/Emotion')#'/import/nobackup_mmv_ioannisp/zs003/face_react/face_v2_newlabel/train/New_Emotion'
        
        # self._emotion_path=os.path.join(self.real_dataset_path, 'Emotion')
            self._3dmm_path = os.path.join(self.real_dataset_path, '3D_FV_files')

        self.mean_face = torch.FloatTensor(
            np.load('./external/FaceVerse/mean_face.npy').astype(np.float32)).view(1, 1, -1)
        self.std_face = torch.FloatTensor(
            np.load('./external/FaceVerse/std_face.npy').astype(np.float32)).view(1, 1, -1)

        self._transform = Transform(img_size, crop_size)
        self._transform_3dmm = transforms.Lambda(lambda e: (e - self.mean_face))

        speaker_path = list(self._list_path.values[:, 1])
        listener_path = list(self._list_path.values[:, 2])
        new_speaker_path = list(self._new_list_path.values[:, 1])
        new_listener_path = list(self._new_list_path.values[:, 2])
        # pdb.set_trace()
        if self._split in ["val","test","train"] or repeat_mirrored:  # training is always mirrored as data augmentation
            speaker_path_tmp = speaker_path + listener_path
            listener_path_tmp = listener_path + speaker_path
            speaker_path = speaker_path_tmp
            listener_path = listener_path_tmp

            new_speaker_path_tmp = new_speaker_path + new_listener_path
            new_listener_path_tmp = new_listener_path + new_speaker_path
            new_speaker_path = new_speaker_path_tmp
            new_listener_path = new_listener_path_tmp

        self.data_list = []
        # pdb.set_trace()
        for i, (sp, lp,ns,nl) in enumerate(zip(speaker_path, listener_path,new_speaker_path,new_listener_path)):
            if 'UDIVA' in sp:
                continue
            # if 'RECOLA' in sp:
            #     # pdb.set_trace()
            #     continue
            # if 'NoXI' in sp:
            #     continue
            ab_speaker_video_path = os.path.join(self._video_path, sp+ '.mp4')
            ab_speaker_audio_path = os.path.join(self._audio_path, sp + '.wav')
            #ab_speaker_emotion_path = os.path.join(self._emotion_path, sp + '.csv')
            ab_speaker_emotion_path = os.path.join(self._emotion_path, ns + '.csv')
            ab_speaker_3dmm_path = os.path.join(self._3dmm_path, sp + '.npy')

            ab_listener_video_path = os.path.join(self._video_path, lp+'.mp4')
            ab_listener_audio_path = os.path.join(self._audio_path, lp + '.wav')
            #ab_listener_emotion_path = os.path.join(self._emotion_path, lp + '.csv')
            ab_listener_emotion_path = os.path.join(self._emotion_path, nl + '.csv')
            ab_listener_3dmm_path = os.path.join(self._3dmm_path, lp + '.npy')
            # pdb.set_trace()
            if 'NoXI' in ab_speaker_emotion_path:
                # pdb.set_trace()
                ab_speaker_emotion_path=ab_speaker_emotion_path.replace('Novice_video','P2')
                ab_speaker_emotion_path=ab_speaker_emotion_path.replace('Expert_video','P1')
            if 'Emotion/RECOLA/group' in ab_speaker_emotion_path:
                # pdb.set_trace()
                ab_speaker_emotion_path=ab_speaker_emotion_path.replace('P25','P1')
                ab_speaker_emotion_path=ab_speaker_emotion_path.replace('P26','P2')
                ab_speaker_emotion_path=ab_speaker_emotion_path.replace('P41','P1')
                ab_speaker_emotion_path=ab_speaker_emotion_path.replace('P42','P2')
                ab_speaker_emotion_path=ab_speaker_emotion_path.replace('P45','P1')
                ab_speaker_emotion_path=ab_speaker_emotion_path.replace('P46','P2')
            if 'NoXI' in ab_listener_emotion_path:
                # pdb.set_trace()
                ab_listener_emotion_path=ab_listener_emotion_path.replace('Novice_video','P2')
                ab_listener_emotion_path=ab_listener_emotion_path.replace('Expert_video','P1')
            if 'Emotion/RECOLA/group' in ab_listener_emotion_path:
                # pdb.set_trace()
                ab_listener_emotion_path=ab_listener_emotion_path.replace('P25','P1')
                ab_listener_emotion_path=ab_listener_emotion_path.replace('P26','P2')
                ab_listener_emotion_path=ab_listener_emotion_path.replace('P41','P1')
                ab_listener_emotion_path=ab_listener_emotion_path.replace('P42','P2')
                ab_listener_emotion_path=ab_listener_emotion_path.replace('P45','P1')
                ab_listener_emotion_path=ab_listener_emotion_path.replace('P46','P2')
            # if 'sbg'  in platform.node() or 'rdg' in platform.node():
            #     ab_speaker_emotion_path=ab_speaker_emotion_path.replace('val/','')
            #     ab_speaker_emotion_path=ab_speaker_emotion_path.replace('test/','')

            #     ab_listener_emotion_path=ab_listener_emotion_path.replace('val/','')
            #     ab_listener_emotion_path=ab_listener_emotion_path.replace('test/','')
            # else:
            # #old
            # ab_speaker_emotion_path=ab_speaker_emotion_path.replace('val','train')
            # ab_speaker_emotion_path=ab_speaker_emotion_path.replace('test','train')

            # ab_listener_emotion_path=ab_listener_emotion_path.replace('val','train')
            # ab_listener_emotion_path=ab_listener_emotion_path.replace('test','train')
            #new
            ab_speaker_video_path=ab_speaker_video_path.replace('face_final_v2/val/Video_files','cropped_face')
            ab_speaker_video_path=ab_speaker_video_path.replace('face_final_v2/train/Video_files','cropped_face')
            ab_speaker_video_path=ab_speaker_video_path.replace('face_final_v2/test/Video_files','cropped_face')

            ab_listener_video_path=ab_listener_video_path.replace('face_final_v2/val/Video_files','cropped_face')
            ab_listener_video_path=ab_listener_video_path.replace('face_final_v2/train/Video_files','cropped_face')
            ab_listener_video_path=ab_listener_video_path.replace('face_final_v2/test/Video_files','cropped_face')#cropped_face #all_originaldata
            # pdb.set_trace()
            # if '053_2016-04-12_Nottingham/P1/4' in ab_speaker_emotion_path:
            #     pdb.set_trace()
            #     a=1
            if not all(map(os.path.exists, [ab_speaker_video_path, ab_speaker_audio_path, ab_speaker_emotion_path, ab_speaker_3dmm_path, ab_listener_video_path, ab_listener_audio_path, ab_listener_emotion_path, ab_listener_3dmm_path])):
                # pdb.set_trace()
                print([ab_speaker_video_path, ab_speaker_audio_path, ab_speaker_emotion_path, ab_speaker_3dmm_path, ab_listener_video_path, ab_listener_audio_path, ab_listener_emotion_path, ab_listener_3dmm_path])
                # pdb.set_trace()
                continue



#------------------------------------------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------------------------------------------

            self.data_list.append(
                {'speaker_video_path': ab_speaker_video_path, 'speaker_audio_path': ab_speaker_audio_path,
                 'speaker_emotion_path': ab_speaker_emotion_path, 'speaker_3dmm_path': ab_speaker_3dmm_path,
                 'listener_video_path': ab_listener_video_path, 'listener_audio_path': ab_listener_audio_path,
                 'listener_emotion_path': ab_listener_emotion_path, 'listener_3dmm_path': ab_listener_3dmm_path})

        self._len = len(self.data_list)
        print(len(self.data_list))

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        # seq_len, fea_dim
        data = self.data_list[index]

        # ========================= Data Augmentation ==========================
        changed_sign = 0
        # if self._split == 'train':  # only done at training time
        #     changed_sign = random.randint(0, 1)

        speaker_prefix = 'speaker' if changed_sign == 0 else 'listener'
        listener_prefix = 'listener' if changed_sign == 0 else 'speaker'

        # ========================= Load Speaker & Listener video clip ==========================
        speaker_video_path = data[f'{speaker_prefix}_video_path']
        listener_video_path = data[f'{listener_prefix}_video_path']

        #if self.load_video_s or self.load_video_l or self.load_ref:  # otherwise, no need to compute these image paths
        # pdb.set_trace()
        with open(speaker_video_path, 'rb') as f:
            vr = VideoReader(f, ctx=cpu(0))
        total_length = len(vr)
        # img_paths = os.listdir(speaker_video_path)
        # total_length = len(img_paths)
        # img_paths = sorted(img_paths, key=cmp_to_key(lambda a, b: int(a[:-4]) - int(b[:-4])))
        # pdb.set_trace()
        cp = random.randint(0, total_length - 1 - self._clip_length*self.interval) if (self._clip_length*self.interval) < total_length else 0
        last_cp= cp if (cp<=self._clip_length*self.interval) else cp-self._clip_length*self.interval
        # cp=0
        # img_paths = img_paths[cp: cp + self._clip_length]

        speaker_video_clip = 0
        if self.load_video_s:
            # clip = []
            # for img_path in img_paths:
            #     img = self._img_loader(os.path.join(speaker_video_path, img_path))
            #     img = self._transform(img)
            #     clip.append(img.unsqueeze(0))
            # speaker_video_clip = torch.cat(clip, dim=0)
            # pdb.set_trace()
            clip = []
            
            # container = av.open(speaker_video_path)  #read mp4 files
            # with open(speaker_video_path, 'rb') as f:
            #     vr = VideoReader(f, ctx=cpu(0))
            for i in range(cp,cp + self._clip_length*self.interval,self.interval):
                # the video reader will handle seeking and skipping in the most efficient manner
                frame = vr[i]
                # print(frame.shape)
                img=Image.fromarray(frame.asnumpy())
                img = self._transform(img)
                clip.append(img.unsqueeze(0))
            speaker_video_clip = torch.cat(clip, dim=0)
        # listener video clip only needed for val/test
        listener_video_clip = 0
        listener_video_clip_last = 0
        # pdb.set_trace()
        if self.load_video_l:
            # clip = []
            # for img_path in img_paths:
            #     img = self._img_loader(os.path.join(listener_video_path, img_path))
            #     img = self._transform(img)
            #     clip.append(img.unsqueeze(0))
            # listener_video_clip = torch.cat(clip, dim=0)
            clip = []
            listener_video_path=listener_video_path.replace('face_final_v2/val/Video_files','cropped_face')
            listener_video_path=listener_video_path.replace('face_final_v2/train/Video_files','cropped_face')
            listener_video_path=listener_video_path.replace('face_final_v2/test/Video_files','cropped_face')
            # container = av.open(speaker_video_path)  #read mp4 files
            with open(listener_video_path, 'rb') as f:
                vr = VideoReader(f, ctx=cpu(0))
            for i in range(cp,cp + self._clip_length*self.interval,self.interval):
            # for i in range(cp,cp + self._clip_length):
                # the video reader will handle seeking and skipping in the most efficient manner
                frame = vr[i]
                # print(frame.shape)
                img=Image.fromarray(frame.asnumpy())
                img = self._transform(img)
                clip.append(img.unsqueeze(0))
            listener_video_clip = torch.cat(clip, dim=0)

            for i in range(last_cp,last_cp + self._clip_length*self.interval,self.interval):
            # for i in range(cp,cp + self._clip_length):
                # the video reader will handle seeking and skipping in the most efficient manner
                frame = vr[i]
                # print(frame.shape)
                img=Image.fromarray(frame.asnumpy())
                img = self._transform(img)
                clip.append(img.unsqueeze(0))
            listener_video_clip_last = torch.cat(clip, dim=0)
            

        # ========================= Load Speaker audio clip (listener audio is NEVER needed) ==========================
        listener_audio_clip, speaker_audio_clip = 0, 0
        speaker_audio_path,listener_audio_path=0, 0

        if self.load_audio:
            speaker_audio_path = data[f'{speaker_prefix}_audio_path']
            speaker_audio_clip = extract_audio_features(speaker_audio_path, self._fps, total_length)
            speaker_audio_clip = speaker_audio_clip[cp:cp + self._clip_length*self.interval:self.interval]
            
            listener_audio_path = data[f'{listener_prefix}_audio_path']
            listener_audio_clip = extract_audio_features(listener_audio_path, self._fps, total_length)
            listener_audio_clip = listener_audio_clip[cp:cp + self._clip_length*self.interval:self.interval]
            
            # listener_audio_path

        # ========================= Load Speaker & Listener emotion ==========================
        listener_emotion, speaker_emotion = 0, 0
        # pdb.set_trace()
        if self.load_emotion_l:
            listener_emotion_path = data[f'{listener_prefix}_emotion_path']
            listener_emotion = pd.read_csv(listener_emotion_path, header=None, delimiter=',')
            listener_emotion = torch.from_numpy(np.array(listener_emotion.drop(0)).astype(np.float32))[
                               cp: cp + self._clip_length*self.interval:self.interval]
            # listener_emotion = torch.from_numpy(np.array(listener_emotion.drop([0,752])).astype(np.float32))[
            #                    cp: cp + self._clip_length*self.interval:self.interval]
        if self.load_emotion_s:
            speaker_emotion_path = data[f'{speaker_prefix}_emotion_path']
            speaker_emotion = pd.read_csv(speaker_emotion_path, header=None, delimiter=',')
            speaker_emotion = torch.from_numpy(np.array(speaker_emotion.drop(0)).astype(np.float32))[
                              cp: cp + self._clip_length*self.interval:self.interval]
            # speaker_emotion = torch.from_numpy(np.array(speaker_emotion.drop([0,752])).astype(np.float32))[
            #                   cp: cp + self._clip_length*self.interval:self.interval]

        # ========================= Load Listener 3DMM ==========================
        listener_3dmm = 0
        if self.load_3dmm_l:
            listener_3dmm_path = data[f'{listener_prefix}_3dmm_path']
            listener_3dmm = torch.FloatTensor(np.load(listener_3dmm_path)).squeeze()
            listener_3dmm = listener_3dmm[cp: cp + self._clip_length*self.interval:self.interval]
            listener_3dmm = self._transform_3dmm(listener_3dmm)[0]

        speaker_3dmm = 0
        if self.load_3dmm_s:
            speaker_3dmm_path = data[f'{speaker_prefix}_3dmm_path']
            speaker_3dmm = torch.FloatTensor(np.load(speaker_3dmm_path)).squeeze()
            speaker_3dmm = speaker_3dmm[cp: cp + self._clip_length*self.interval:self.interval]
            speaker_3dmm = self._transform_3dmm(speaker_3dmm)[0]

        # ========================= Load Listener Reference ==========================
        listener_reference = 0
        if self.load_ref:
            # img_paths = os.listdir(listener_video_path)
            # img_paths = sorted(img_paths, key=cmp_to_key(lambda a, b: int(a[:-4]) - int(b[:-4])))
            # listener_reference = self._img_loader(os.path.join(listener_video_path, img_paths[0]))
            # listener_reference = self._transform(listener_reference)
            listener_video_path=listener_video_path.replace('face_final_v2/val/Video_files','cropped_face')
            listener_video_path=listener_video_path.replace('face_final_v2/train/Video_files','cropped_face')
            listener_video_path=listener_video_path.replace('face_final_v2/test/Video_files','cropped_face')
            container = av.open(listener_video_path)  #read mp4 files
            # pdb.set_trace()
            for frame in container.decode(video=0):
                img=frame.to_image().convert('RGB')
                # img = self._transform(img)
                break
            listener_reference = self._transform(img)
        # return speaker_video_clip, speaker_audio_clip, speaker_emotion, speaker_3dmm, listener_video_clip, listener_audio_clip, listener_emotion, listener_3dmm, listener_reference
        # pdb.set_trace()
        return {'img': speaker_video_clip, 'index': index,'speaker_audio_clip': speaker_audio_clip,'speaker_emotion': speaker_emotion,
                                'speaker_3dmm': speaker_3dmm,'speaker_audio_path':speaker_audio_path,
                                'listener_video_clip': listener_video_clip, 'listener_video_clip_last':listener_video_clip_last,
                                'listener_emotion': listener_emotion,
                                'listener_3dmm': listener_3dmm,'listener_audio_path':listener_audio_path,
                                'listener_reference': listener_reference,
                                'speaker_emotion_path':speaker_emotion_path,'listener_emotion_path':listener_emotion_path,
                                'cp':[cp,cp + self._clip_length*self.interval]}
    def __len__(self):
        return self._len


class ReactionDataset_readvideo_twocrop(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self,root_path, real_dataset_path, split, img_size=256, crop_size=224, clip_length=4, fps=25,
                 load_audio=True, load_video_s=True, load_video_l=True, load_emotion_s=False, load_emotion_l=False,
                 load_3dmm_s=False, load_3dmm_l=False, load_ref=True,
                 repeat_mirrored=True,interval=5):
        """
        Args:
            root_path: (str) Path to the data folder.
            split: (str) 'train' or 'val' or 'test' split.
            img_size: (int) Size of the image.
            crop_size: (int) Size of the crop.
            clip_length: (int) Number of frames in a clip.
            fps: (int) Frame rate of the video.
            load_audio: (bool) Whether to load audio features.
            load_video_s: (bool) Whether to load speaker video features.
            load_video_l: (bool) Whether to load listener video features.
            load_emotion: (bool) Whether to load emotion labels.
            load_3dmm: (bool) Whether to load 3DMM parameters.
            repeat_mirrored: (bool) Whether to extend dataset with mirrored speaker/listener. This is used for val/test.
        """

        self._root_path = root_path
        self._img_loader = pil_loader
        self._clip_length = clip_length
        self.interval=interval
        self._fps = fps
        self._split = split

        self._data_path = os.path.join(self._root_path, self._split)
        if 'sbg'  in platform.node() or 'rdg' in platform.node():
            self.real_dataset_path=real_dataset_path
        else:
            self.real_dataset_path=os.path.join(real_dataset_path, self._split)
        self._list_path = pd.read_csv(os.path.join(self._root_path, self._split + '.csv'), header=None, delimiter=',')
        
        self._new_list_path = self._list_path.copy()#pd.read_csv(os.path.join(self._root_path, self._split + '_new.csv'), header=None, delimiter=',')
        self._list_path = self._list_path.drop(0)
        self._new_list_path = self._new_list_path.drop(0)

        self.load_audio = load_audio
        self.load_video_s = load_video_s
        self.load_video_l = load_video_l
        self.load_3dmm_s = load_3dmm_s
        self.load_3dmm_l = load_3dmm_l
        self.load_emotion_s = load_emotion_s
        self.load_emotion_l = load_emotion_l
        self.load_ref = load_ref

        # self.real_dataset_path=real_dataset_path
        #
        # self._audio_path = os.path.join(self._data_path, 'Audio_files')
        # self._video_path = os.path.join(self._data_path, 'Video_files')
        # self._emotion_path = os.path.join(self._data_path, 'Emotion')
        # self._3dmm_path = os.path.join(self._data_path, '3D_FV_files')
        # self.reaction_filepath24='/import/nobackup_mmv_ioannisp/zs003/react_2024'#include emotion and data matrix
        if 'sbg'  in platform.node() or 'dg' in platform.node():
            # self._audio_path = os.path.join('/data/scratch/acw569/face_react/from_hengde/', self._split+'/Audio_files')#'/data/scratch/acw569/face_react/audios'#os.path.join(self.real_dataset_path, 'Audio_files')
            # # self._video_path = os.path.join('/data/scratch/acw569/face_react/from_hengde/', self._split+'/Video_files')#os.path.join(self.real_dataset_path, 'aligned_faces')
            # self._video_path = '/data/scratch/acw569/face_react/new_crop'
            # self.aligned_video_path=os.path.join('/data/scratch/acw569/face_react/from_hengde/', self._split+'/Video_files')
            # self._emotion_path = os.path.join('/data/scratch/acw569/face_react/from_hengde/', self._split+'/Emotion')#'/data/scratch/acw569/face_react/New_Emotion'
            # self._3dmm_path = os.path.join('/data/scratch/acw569/face_react/from_hengde/', self._split+'/3D_FV_files')#'/data/scratch/acw569/face_react/3dmm_param_all/3DMM_param'
            self._audio_path = os.path.join('/data/scratch/acw569/face_react/new_reaction_2024data', self._split+'/Audio_files')#'/data/scratch/acw569/face_react/audios'#os.path.join(self.real_dataset_path, 'Audio_files')
            # self._video_path = os.path.join('/data/scratch/acw569/face_react/from_hengde/', self._split+'/Video_files')#os.path.join(self.real_dataset_path, 'aligned_faces')
            
            self._video_path = '/data/scratch/acw569/face_react/new_reaction_2024data/cropped_face'
            self.aligned_video_path=self._video_path
            self._emotion_path = os.path.join('/data/scratch/acw569/face_react/new_reaction_2024data', self._split+'/Emotion')#'/data/scratch/acw569/face_react/New_Emotion'
            self._3dmm_path = os.path.join('/data/scratch/acw569/face_react/new_reaction_2024data', self._split+'/3D_FV_files')#'/data/scratch/acw569/face_react/3dmm_param_all/3DMM_param'
        
        else:
            self.emo_filepath24='/import/nobackup_mmv_ioannisp/zs003/react_2024'#include emotion and data matrix
        
            self._audio_path = '/import/nobackup_mmv_ioannisp/zs003/face_react/new_audio_folder/all_together'#os.path.join(self.real_dataset_path, 'Audio_files')
            self._video_path = os.path.join(self.real_dataset_path, 'Video_files')
            # self._video_path='/import/nobackup_mmv_ioannisp/zs003/face_react/NOXI_original/'
            self._emotion_path = '/import/nobackup_mmv_ioannisp/zs003/face_react/face_v2_newlabel/train/New_Emotion'
            # pdb.set_trace()
            self._emotion_path = os.path.join(self.emo_filepath24, self._split+'/Emotion')#'/import/nobackup_mmv_ioannisp/zs003/face_react/face_v2_newlabel/train/New_Emotion'
        
        # self._emotion_path=os.path.join(self.real_dataset_path, 'Emotion')
            self._3dmm_path = os.path.join(self.real_dataset_path, '3D_FV_files')

        self.mean_face = torch.FloatTensor(
            np.load('./external/FaceVerse/mean_face.npy').astype(np.float32)).view(1, 1, -1)
        self.std_face = torch.FloatTensor(
            np.load('./external/FaceVerse/std_face.npy').astype(np.float32)).view(1, 1, -1)

        self._transform = Transform(img_size, crop_size)
        self._transform_3dmm = transforms.Lambda(lambda e: (e - self.mean_face))

        speaker_path = list(self._list_path.values[:, 1])
        listener_path = list(self._list_path.values[:, 2])
        new_speaker_path = list(self._new_list_path.values[:, 1])
        new_listener_path = list(self._new_list_path.values[:, 2])
        # pdb.set_trace()
        if self._split in ["val","test","train"] or repeat_mirrored:  # training is always mirrored as data augmentation
            speaker_path_tmp = speaker_path + listener_path
            listener_path_tmp = listener_path + speaker_path
            speaker_path = speaker_path_tmp
            listener_path = listener_path_tmp

            new_speaker_path_tmp = new_speaker_path + new_listener_path
            new_listener_path_tmp = new_listener_path + new_speaker_path
            new_speaker_path = new_speaker_path_tmp
            new_listener_path = new_listener_path_tmp

        self.data_list = []
        # pdb.set_trace()
        for i, (sp, lp,ns,nl) in enumerate(zip(speaker_path, listener_path,new_speaker_path,new_listener_path)):
            if 'UDIVA' in sp:
                continue
            # if 'RECOLA' in sp:
            #     # pdb.set_trace()
            #     continue
            # if 'NoXI' in sp:
            #     continue
            ab_speaker_video_path = os.path.join(self._video_path, sp+ '.mp4')
            
            ab_speaker_audio_path = os.path.join(self._audio_path, sp + '.wav')
            #ab_speaker_emotion_path = os.path.join(self._emotion_path, sp + '.csv')
            ab_speaker_emotion_path = os.path.join(self._emotion_path, ns + '.csv')
            ab_speaker_3dmm_path = os.path.join(self._3dmm_path, sp + '.npy')

            #aligned crop
            ab_speaker_aligned_video_path=os.path.join(self.aligned_video_path,sp+'.mp4')

            ab_listener_video_path = os.path.join(self._video_path, lp+'.mp4')
            ab_listener_audio_path = os.path.join(self._audio_path, lp + '.wav')
            #ab_listener_emotion_path = os.path.join(self._emotion_path, lp + '.csv')
            ab_listener_emotion_path = os.path.join(self._emotion_path, nl + '.csv')
            ab_listener_3dmm_path = os.path.join(self._3dmm_path, lp + '.npy')
    
            ab_listener_aligned_video_path=os.path.join(self.aligned_video_path,sp+'.mp4')
            # pdb.set_trace()
            if 'NoXI' in ab_speaker_emotion_path:
                # pdb.set_trace()
                ab_speaker_emotion_path=ab_speaker_emotion_path.replace('Novice_video','P2')
                ab_speaker_emotion_path=ab_speaker_emotion_path.replace('Expert_video','P1')
            if 'Emotion/RECOLA/group' in ab_speaker_emotion_path:
                # pdb.set_trace()
                ab_speaker_emotion_path=ab_speaker_emotion_path.replace('P25','P1')
                ab_speaker_emotion_path=ab_speaker_emotion_path.replace('P26','P2')
                ab_speaker_emotion_path=ab_speaker_emotion_path.replace('P41','P1')
                ab_speaker_emotion_path=ab_speaker_emotion_path.replace('P42','P2')
                ab_speaker_emotion_path=ab_speaker_emotion_path.replace('P45','P1')
                ab_speaker_emotion_path=ab_speaker_emotion_path.replace('P46','P2')
            if 'NoXI' in ab_listener_emotion_path:
                # pdb.set_trace()
                ab_listener_emotion_path=ab_listener_emotion_path.replace('Novice_video','P2')
                ab_listener_emotion_path=ab_listener_emotion_path.replace('Expert_video','P1')
            if 'Emotion/RECOLA/group' in ab_listener_emotion_path:
                # pdb.set_trace()
                ab_listener_emotion_path=ab_listener_emotion_path.replace('P25','P1')
                ab_listener_emotion_path=ab_listener_emotion_path.replace('P26','P2')
                ab_listener_emotion_path=ab_listener_emotion_path.replace('P41','P1')
                ab_listener_emotion_path=ab_listener_emotion_path.replace('P42','P2')
                ab_listener_emotion_path=ab_listener_emotion_path.replace('P45','P1')
                ab_listener_emotion_path=ab_listener_emotion_path.replace('P46','P2')
            # if 'sbg'  in platform.node() or 'rdg' in platform.node():
            #     ab_speaker_emotion_path=ab_speaker_emotion_path.replace('val/','')
            #     ab_speaker_emotion_path=ab_speaker_emotion_path.replace('test/','')

            #     ab_listener_emotion_path=ab_listener_emotion_path.replace('val/','')
            #     ab_listener_emotion_path=ab_listener_emotion_path.replace('test/','')
            # else:
            # #old
            # ab_speaker_emotion_path=ab_speaker_emotion_path.replace('val','train')
            # ab_speaker_emotion_path=ab_speaker_emotion_path.replace('test','train')

            # ab_listener_emotion_path=ab_listener_emotion_path.replace('val','train')
            # ab_listener_emotion_path=ab_listener_emotion_path.replace('test','train')
            #new
            ab_speaker_video_path=ab_speaker_video_path.replace('face_final_v2/val/Video_files','cropped_face')
            ab_speaker_video_path=ab_speaker_video_path.replace('face_final_v2/train/Video_files','cropped_face')
            ab_speaker_video_path=ab_speaker_video_path.replace('face_final_v2/test/Video_files','cropped_face')
            
            ab_speaker_aligned_video_path=ab_speaker_aligned_video_path.replace('face_final_v2/val/Video_files','cropped_face')
            ab_speaker_aligned_video_path=ab_speaker_aligned_video_path.replace('face_final_v2/train/Video_files','cropped_face')
            ab_speaker_aligned_video_path=ab_speaker_aligned_video_path.replace('face_final_v2/test/Video_files','cropped_face')
            
            

            ab_listener_video_path=ab_listener_video_path.replace('face_final_v2/val/Video_files','cropped_face')
            ab_listener_video_path=ab_listener_video_path.replace('face_final_v2/train/Video_files','cropped_face')
            ab_listener_video_path=ab_listener_video_path.replace('face_final_v2/test/Video_files','cropped_face')#cropped_face #all_originaldata
            
            ab_listener_aligned_video_path=ab_listener_aligned_video_path.replace('face_final_v2/val/Video_files','cropped_face')
            ab_listener_aligned_video_path=ab_listener_aligned_video_path.replace('face_final_v2/train/Video_files','cropped_face')
            ab_listener_aligned_video_path=ab_listener_aligned_video_path.replace('face_final_v2/test/Video_files','cropped_face')#cropped_face #all_originaldata
            
            # pdb.set_trace()
            # if '053_2016-04-12_Nottingham/P1/4' in ab_speaker_emotion_path:
            #     pdb.set_trace()
            #     a=1
            if not all(map(os.path.exists, [ab_speaker_video_path, ab_speaker_audio_path, ab_speaker_emotion_path, ab_speaker_3dmm_path, ab_listener_video_path, ab_listener_audio_path, ab_listener_emotion_path, ab_listener_3dmm_path])):
                # pdb.set_trace()
                print([ab_speaker_video_path, ab_speaker_audio_path, ab_speaker_emotion_path, ab_speaker_3dmm_path, ab_listener_video_path, ab_listener_audio_path, ab_listener_emotion_path, ab_listener_3dmm_path])
                # pdb.set_trace()
                continue



#------------------------------------------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------------------------------------------

            self.data_list.append(
                {'speaker_video_path': ab_speaker_video_path, 'speaker_aligned_video_path':ab_speaker_aligned_video_path,'speaker_audio_path': ab_speaker_audio_path,
                 'speaker_emotion_path': ab_speaker_emotion_path, 'speaker_3dmm_path': ab_speaker_3dmm_path,
                 'listener_video_path': ab_listener_video_path, 'listener_aligned_video_path':ab_listener_aligned_video_path,'listener_audio_path': ab_listener_audio_path,
                 'listener_emotion_path': ab_listener_emotion_path, 'listener_3dmm_path': ab_listener_3dmm_path})

        self._len = len(self.data_list)
        print(len(self.data_list))

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        # seq_len, fea_dim
        data = self.data_list[index]

        # ========================= Data Augmentation ==========================
        changed_sign = 0
        # if self._split == 'train':  # only done at training time
        #     changed_sign = random.randint(0, 1)

        speaker_prefix = 'speaker' if changed_sign == 0 else 'listener'
        listener_prefix = 'listener' if changed_sign == 0 else 'speaker'

        # ========================= Load Speaker & Listener video clip ==========================
        speaker_video_path = data[f'{speaker_prefix}_video_path']
        listener_video_path = data[f'{listener_prefix}_video_path']

        speaker_align_video_path = data[f'{speaker_prefix}_aligned_video_path']
        listener_align_video_path = data[f'{listener_prefix}_aligned_video_path']
        #if self.load_video_s or self.load_video_l or self.load_ref:  # otherwise, no need to compute these image paths
        # pdb.set_trace()
        with open(speaker_video_path, 'rb') as f:
            vr = VideoReader(f, ctx=cpu(0))
        total_length = len(vr)
        # img_paths = os.listdir(speaker_video_path)
        # total_length = len(img_paths)
        # img_paths = sorted(img_paths, key=cmp_to_key(lambda a, b: int(a[:-4]) - int(b[:-4])))
        # pdb.set_trace()
        cp = random.randint(0, total_length - 1 - self._clip_length*self.interval) if (self._clip_length*self.interval) < total_length else 0
        # cp=0
        # img_paths = img_paths[cp: cp + self._clip_length]

        speaker_video_clip = 0
        if self.load_video_s:
            # clip = []
            # for img_path in img_paths:
            #     img = self._img_loader(os.path.join(speaker_video_path, img_path))
            #     img = self._transform(img)
            #     clip.append(img.unsqueeze(0))
            # speaker_video_clip = torch.cat(clip, dim=0)
            # pdb.set_trace()
            clip = []
            
            # container = av.open(speaker_video_path)  #read mp4 files
            # with open(speaker_video_path, 'rb') as f:
            #     vr = VideoReader(f, ctx=cpu(0))
            for i in range(cp,cp + self._clip_length*self.interval,self.interval):
                # the video reader will handle seeking and skipping in the most efficient manner
                frame = vr[i]
                # print(frame.shape)
                img=Image.fromarray(frame.asnumpy())
                img = self._transform(img)
                clip.append(img.unsqueeze(0))
            speaker_video_clip = torch.cat(clip, dim=0)
        
        with open(speaker_align_video_path, 'rb') as f:
            vr = VideoReader(f, ctx=cpu(0))
        total_length = len(vr)
        # img_paths = os.listdir(speaker_video_path)
        # total_length = len(img_paths)
        # img_paths = sorted(img_paths, key=cmp_to_key(lambda a, b: int(a[:-4]) - int(b[:-4])))
        # pdb.set_trace()
        cp = random.randint(0, total_length - 1 - self._clip_length*self.interval) if (self._clip_length*self.interval) < total_length else 0
        last_cp= cp if (cp<=self._clip_length*self.interval) else cp-self._clip_length*self.interval

        # cp=0
        # img_paths = img_paths[cp: cp + self._clip_length]

        speaker_align_video_clip = 0
        if self.load_video_s:
            # clip = []
            # for img_path in img_paths:
            #     img = self._img_loader(os.path.join(speaker_video_path, img_path))
            #     img = self._transform(img)
            #     clip.append(img.unsqueeze(0))
            # speaker_video_clip = torch.cat(clip, dim=0)
            # pdb.set_trace()
            clip = []
            
            # container = av.open(speaker_video_path)  #read mp4 files
            # with open(speaker_video_path, 'rb') as f:
            #     vr = VideoReader(f, ctx=cpu(0))
            for i in range(cp,cp + self._clip_length*self.interval,self.interval):
                # the video reader will handle seeking and skipping in the most efficient manner
                frame = vr[i]
                # print(frame.shape)
                img=Image.fromarray(frame.asnumpy())
                img = self._transform(img)
                clip.append(img.unsqueeze(0))
            speaker_align_video_clip = torch.cat(clip, dim=0)
        # listener video clip only needed for val/test
        listener_video_clip = 0
        if self.load_video_l:
            # clip = []
            # for img_path in img_paths:
            #     img = self._img_loader(os.path.join(listener_video_path, img_path))
            #     img = self._transform(img)
            #     clip.append(img.unsqueeze(0))
            # listener_video_clip = torch.cat(clip, dim=0)
            clip = []
            listener_video_path=listener_video_path.replace('face_final_v2/val/Video_files','cropped_face')
            listener_video_path=listener_video_path.replace('face_final_v2/train/Video_files','cropped_face')
            listener_video_path=listener_video_path.replace('face_final_v2/test/Video_files','cropped_face')
            # container = av.open(speaker_video_path)  #read mp4 files
            with open(listener_video_path, 'rb') as f:
                vr = VideoReader(f, ctx=cpu(0))
            for i in range(cp,cp + self._clip_length*self.interval,self.interval):
            # for i in range(cp,cp + self._clip_length):
                # the video reader will handle seeking and skipping in the most efficient manner
                frame = vr[i]
                # print(frame.shape)
                img=Image.fromarray(frame.asnumpy())
                img = self._transform(img)
                clip.append(img.unsqueeze(0))
            listener_video_clip = torch.cat(clip, dim=0)

            for i in range(last_cp,last_cp + self._clip_length*self.interval,self.interval):
            # for i in range(cp,cp + self._clip_length):
                # the video reader will handle seeking and skipping in the most efficient manner
                frame = vr[i]
                # print(frame.shape)
                img=Image.fromarray(frame.asnumpy())
                img = self._transform(img)
                clip.append(img.unsqueeze(0))
            listener_video_clip_last = torch.cat(clip, dim=0)
        # pdb.set_trace()
        listener_align_video_clip = 0
        if self.load_video_l:
            # clip = []
            # for img_path in img_paths:
            #     img = self._img_loader(os.path.join(listener_video_path, img_path))
            #     img = self._transform(img)
            #     clip.append(img.unsqueeze(0))
            # listener_video_clip = torch.cat(clip, dim=0)
            clip = []
            listener_align_video_path=listener_align_video_path.replace('face_final_v2/val/Video_files','cropped_face')
            listener_align_video_path=listener_align_video_path.replace('face_final_v2/train/Video_files','cropped_face')
            listener_align_video_path=listener_align_video_path.replace('face_final_v2/test/Video_files','cropped_face')
            # container = av.open(speaker_video_path)  #read mp4 files
            with open(listener_video_path, 'rb') as f:
                vr = VideoReader(f, ctx=cpu(0))
            for i in range(cp,cp + self._clip_length*self.interval,self.interval):
            # for i in range(cp,cp + self._clip_length):
                # the video reader will handle seeking and skipping in the most efficient manner
                frame = vr[i]
                # print(frame.shape)
                img=Image.fromarray(frame.asnumpy())
                img = self._transform(img)
                clip.append(img.unsqueeze(0))
            listener_align_video_clip = torch.cat(clip, dim=0)

        # ========================= Load Speaker audio clip (listener audio is NEVER needed) ==========================
        listener_audio_clip, speaker_audio_clip = 0, 0
        if self.load_audio:
            speaker_audio_path = data[f'{speaker_prefix}_audio_path']
            speaker_audio_clip = extract_audio_features(speaker_audio_path, self._fps, total_length)
            speaker_audio_clip = speaker_audio_clip[cp:cp + self._clip_length*self.interval:self.interval]


        # ========================= Load Speaker & Listener emotion ==========================
        listener_emotion, speaker_emotion = 0, 0
        # pdb.set_trace()
        if self.load_emotion_l:
            listener_emotion_path = data[f'{listener_prefix}_emotion_path']
            listener_emotion = pd.read_csv(listener_emotion_path, header=None, delimiter=',')
            listener_emotion = torch.from_numpy(np.array(listener_emotion.drop(0)).astype(np.float32))[
                               cp: cp + self._clip_length*self.interval:self.interval]
            # listener_emotion = torch.from_numpy(np.array(listener_emotion.drop([0,752])).astype(np.float32))[
            #                    cp: cp + self._clip_length*self.interval:self.interval]
        if self.load_emotion_s:
            speaker_emotion_path = data[f'{speaker_prefix}_emotion_path']
            speaker_emotion = pd.read_csv(speaker_emotion_path, header=None, delimiter=',')
            speaker_emotion = torch.from_numpy(np.array(speaker_emotion.drop(0)).astype(np.float32))[
                              cp: cp + self._clip_length*self.interval:self.interval]
            # speaker_emotion = torch.from_numpy(np.array(speaker_emotion.drop([0,752])).astype(np.float32))[
            #                   cp: cp + self._clip_length*self.interval:self.interval]

        # ========================= Load Listener 3DMM ==========================
        listener_3dmm = 0
        if self.load_3dmm_l:
            listener_3dmm_path = data[f'{listener_prefix}_3dmm_path']
            listener_3dmm = torch.FloatTensor(np.load(listener_3dmm_path)).squeeze()
            listener_3dmm = listener_3dmm[cp: cp + self._clip_length*self.interval:self.interval]
            listener_3dmm = self._transform_3dmm(listener_3dmm)[0]

        speaker_3dmm = 0
        if self.load_3dmm_s:
            speaker_3dmm_path = data[f'{speaker_prefix}_3dmm_path']
            speaker_3dmm = torch.FloatTensor(np.load(speaker_3dmm_path)).squeeze()
            speaker_3dmm = speaker_3dmm[cp: cp + self._clip_length*self.interval:self.interval]
            speaker_3dmm = self._transform_3dmm(speaker_3dmm)[0]

        # ========================= Load Listener Reference ==========================
        listener_reference = 0
        if self.load_ref:
            # img_paths = os.listdir(listener_video_path)
            # img_paths = sorted(img_paths, key=cmp_to_key(lambda a, b: int(a[:-4]) - int(b[:-4])))
            # listener_reference = self._img_loader(os.path.join(listener_video_path, img_paths[0]))
            # listener_reference = self._transform(listener_reference)
            listener_video_path=listener_video_path.replace('face_final_v2/val/Video_files','cropped_face')
            listener_video_path=listener_video_path.replace('face_final_v2/train/Video_files','cropped_face')
            listener_video_path=listener_video_path.replace('face_final_v2/test/Video_files','cropped_face')
            container = av.open(listener_video_path)  #read mp4 files
            # pdb.set_trace()
            for frame in container.decode(video=0):
                img=frame.to_image().convert('RGB')
                # img = self._transform(img)
                break
            listener_reference = self._transform(img)
        # return speaker_video_clip, speaker_audio_clip, speaker_emotion, speaker_3dmm, listener_video_clip, listener_audio_clip, listener_emotion, listener_3dmm, listener_reference
        # pdb.set_trace()
        return {'img': speaker_video_clip,'speaker_align_video_clip':speaker_align_video_clip, 'index': index,'speaker_audio_clip': speaker_audio_clip,'speaker_emotion': speaker_emotion,
                                'speaker_3dmm': speaker_3dmm,'listener_video_clip': listener_video_clip,'listener_align_video_clip':listener_align_video_clip,
                                'listener_video_clip_last':listener_video_clip_last,'listener_emotion': listener_emotion,
                                'listener_3dmm': listener_3dmm,'listener_reference': listener_reference,
                                'speaker_emotion_path':speaker_emotion_path,'listener_emotion_path':listener_emotion_path,
                                'cp':[cp,cp + self._clip_length*self.interval]}
    def __len__(self):
        return self._len

class ReactionDataset_2025(data.Dataset):
    def __init__(self,
                 root_dir: str = './data',  # /path/to/dataset
                 split: str = 'train',
                 clip_length: int = None,
                 target_size: int = 224,
                 crop_size: int = 224,
                 fps: int = 30,
                 audio_feature_type: str = 'mfcc',
                 load_video_s: bool = False,
                 load_video_l: bool = False,
                 load_audio: bool = True,
                 load_emotion_s: bool = True,
                 load_emotion_l: bool = True,
                 load_3dmm_s: bool = True,
                 load_3dmm_l: bool = True,
                 normalize_3dmm: str = 'standard',  # standard | zero_center
                 interval=5,
                 **kwargs,
                 ):

        self._root_dir = root_dir
        self._clip_length = clip_length
        self._fps = fps
        self._split = split
        self.load_video_s = load_video_s
        self.load_video_l = load_video_l
        self.load_audio = load_audio
        self.load_emotion_s = load_emotion_s
        self.load_emotion_l = load_emotion_l
        self.load_3dmm_s = load_3dmm_s
        self.load_3dmm_l = load_3dmm_l
        self.interval=interval

        dataset_dir = os.path.join(root_dir, self._split)
        self.audio_feature_type = audio_feature_type
        if self.audio_feature_type == 'wav2vec':
            self._audio_dir = os.path.join(dataset_dir, 'audio-features')
        elif self.audio_feature_type == 'mfcc':
            self._audio_dir = os.path.join(dataset_dir, 'audio')
        # self._video_dir = os.path.join(dataset_dir, 'video-face-crop')#original crop
        self._video_dir = os.path.join(dataset_dir, 'ZL_crop_test')     #face detection crop, retinaface
        self._emotion_dir = os.path.join(dataset_dir, 'facial-attributes')
        self._3dmm_dir = os.path.join(dataset_dir, 'coefficients')

        gt_path_dict = {}
        for root, _, files in os.walk(self._video_dir):
            for path in files:
                path = Path(path)
                file, ext = path.stem, path.suffix

                if ext.lower() != '.mp4':
                    continue

                session_id = Path(*Path(root).parts[-2:]) # listener/session0

                file_path = session_id / file
                if session_id not in gt_path_dict:
                    gt_path_dict[session_id] = [file_path]
                else:
                    gt_path_dict[session_id].append(file_path)

        speaker_path_list = []
        listener_path_list = []
        gt_path_list = []

        for root, _, files in os.walk(self._video_dir):
            for path in files:
                path = Path(path)
                file, ext = path.stem, path.suffix
                # file, ext = os.path.splitext(path)
                # e.g., 'Camera-2024-06-21-103121-103102', '.mp4'
                if ext.lower() != '.mp4':
                    continue

                #  role: listener | speaker
                #  session id: session*
                #  file: Camera-2024-06-21-103121-103102
                parts = Path(root).parts
                file_path = Path(*parts[-2:]) / file
                role = parts[-2]
                session_id = Path(parts[-1])
                gt_session_id = 'speaker' / session_id if role == 'listener' else 'listener' / session_id
                listener_file_path = gt_session_id / file

                speaker_path_list.append(file_path)
                listener_path_list.append(listener_file_path)
                listener_gt_paths = gt_path_dict[gt_session_id]
                gt_path_list.append(listener_gt_paths)

        self.speaker_path_list = speaker_path_list.copy()
        self.listener_path_list = listener_path_list.copy()
        self.gt_path_list = gt_path_list.copy()

        # mean_face_path = os.path.join(hydra.utils.get_original_cwd(), 'external/FaceVerse/mean_face.npy')
        # self.mean_face = torch.FloatTensor(
        #     np.load(mean_face_path).astype(np.float32)).view(1, 1, -1)
        # std_face_path = os.path.join(hydra.utils.get_original_cwd(), 'external/FaceVerse/std_face.npy')
        # self.std_face = torch.FloatTensor(
        #     np.load(std_face_path).astype(np.float32)).view(1, 1, -1)

        mean_face_path = os.path.join(os.getcwd(), 'external/FaceVerse/mean_face.npy')
        self.mean_face = torch.FloatTensor(
            np.load(mean_face_path).astype(np.float32)).view(1, 1, -1)
        std_face_path = os.path.join(os.getcwd(), 'external/FaceVerse/std_face.npy')
        self.std_face = torch.FloatTensor(
            np.load(std_face_path).astype(np.float32)).view(1, 1, -1)


        
        self.normalize_3dmm = normalize_3dmm
        if normalize_3dmm == 'standard':
            self._transform_3dmm = transforms.Lambda(lambda e: (e - self.mean_face) / self.std_face)
        elif normalize_3dmm == 'zero_center':
            self._transform_3dmm = transforms.Lambda(lambda e: e - self.mean_face)
        else:
            raise ValueError(f"Unknown normalize_3dmm: {normalize_3dmm}")

        self._transform = Transform(target_size, crop_size)
        self._len = len(self.speaker_path_list)
        print('reaction dataset, length is', self._len )
    def __getitem__(self, index):
        speaker_path = self.speaker_path_list[index]  # e.g., speaker/session*/Camera-2024-06-21-103121-103102
        listener_path = self.listener_path_list[index]  # e.g., listener/session*/Camera-2024-06-21-103121-103102

        video_path = os.fspath(Path(self._video_dir) / speaker_path.with_suffix('.mp4'))
        vr = VideoReader(video_path, ctx=cpu(0))
        total_length = len(vr)  # length from 58 to more than 20000
        if not self.load_video_s:
            del vr

        if self._split == "test":
            k_select = 9
            gt_paths = self.gt_path_list[index]
            listener_paths = []
            listener_paths.append(listener_path)
            c = gt_paths if len(gt_paths) <= 1 else [p for p in gt_paths if p != listener_path]
            listener_paths.extend(random.sample(c, k_select) \
                                      if len(c) >= k_select else random.choices(c, k=k_select))
            self._clip_length = total_length

        # cp = random.randint(0, total_length - self._clip_length) \
        #     if total_length > self._clip_length else 0
        cp = random.randint(0, total_length - 1 - self._clip_length*self.interval) if (self._clip_length*self.interval) < total_length else 0
        last_cp= cp if (cp<=self._clip_length*self.interval) else cp-self._clip_length*self.interval
        # ========== Load Speaker Data ==========
        # speaker's face video clip
        speaker_video_clip = torch.zeros(size=(0,))
        if self.load_video_s:
            clip = []
            for i in range(cp,cp + self._clip_length*self.interval,self.interval):
                if i >= len(vr):
                    break
                frame = vr[i]
                img = Image.fromarray(frame.asnumpy())
                img = self._transform(img)
                clip.append(img)
            del vr
            speaker_video_clip = torch.stack(clip, dim=0)
            # if total_length < self._clip_length*self.interval or len(speaker_video_clip)<self._clip_length:
            if len(speaker_video_clip)<self._clip_length:

                speaker_video_clip = torch.cat((speaker_video_clip,
                                                torch.zeros(size=(self._clip_length - len(speaker_video_clip),
                                                                  *speaker_video_clip.shape[1:]))), dim=0)

        # speaker's facial attribute (emotion) clip
        speaker_emotion_clip = torch.zeros(size=(0,))
        if self.load_emotion_s:
            emotion_path = os.fspath(Path(self._emotion_dir) / speaker_path.with_suffix('.npy'))
            emotion = np.load(emotion_path)
            speaker_emotion_clip = torch.from_numpy(emotion)
            speaker_emotion_clip = speaker_emotion_clip[cp: cp + self._clip_length*self.interval:self.interval]  # (25-d)
            if len(speaker_emotion_clip)<self._clip_length:
                speaker_emotion_clip=torch.cat((speaker_emotion_clip,torch.zeros(self._clip_length-len(speaker_emotion_clip),speaker_emotion_clip.shape[-1])),dim=0)
        # speaker's 3DMM coefficient (facial motion) clip
        speaker_params_clip = torch.zeros(size=(0,))
        if self.load_3dmm_s:
            params_path = os.fspath(Path(self._3dmm_dir) / speaker_path.with_suffix('.npy'))
            params = torch.FloatTensor(np.load(params_path)).squeeze()
            params = params[cp: cp + self._clip_length*self.interval:self.interval]
            speaker_params_clip = self._transform_3dmm(params)[0]  # (58-d)

        # speaker's audio feature clip
        speaker_audio_clip = torch.zeros(size=(0,))
        if self.load_audio:
            if self.audio_feature_type == 'wav2vec':
                audio_path = os.fspath(Path(self._audio_dir) / speaker_path.with_suffix('.npy'))
                speaker_audio_clip = np.load(audio_path)  # (768-d)
                #
                listener_audio_path=os.fspath(Path(self._audio_dir) / listener_path.with_suffix('.npy'))
            elif self.audio_feature_type == 'mfcc':
                audio_path = os.fspath(Path(self._audio_dir) / speaker_path.with_suffix('.wav'))
                speaker_audio_clip = extract_audio_features(audio_path, self._fps, total_length)  # (78-d)
                #load listener audio
                listener_audio_path=os.fspath(Path(self._audio_dir) / listener_path.with_suffix('.wav'))
            else:
                raise ValueError(f"Unknown audio feature type: {self.audio_feature_type}")
            speaker_audio_clip = torch.from_numpy(speaker_audio_clip)[cp: cp + self._clip_length*self.interval:self.interval]
            speaker_audio_path=audio_path
            listener_audio_path=listener_audio_path

        # if total_length < self._clip_length*self.interval or len(speaker_audio_clip)<self._clip_length:
        if len(speaker_audio_clip)<self._clip_length or len(speaker_emotion_clip)<self._clip_length or len(speaker_params_clip)<self._clip_length:
            speaker_audio_clip = torch.cat((speaker_audio_clip,
                                            torch.zeros(size=(self._clip_length - len(speaker_audio_clip),
                                                              speaker_audio_clip.shape[-1]))), dim=0) \
                if self.load_audio else speaker_audio_clip
            speaker_emotion_clip = torch.cat((speaker_emotion_clip,
                                            torch.zeros(size=(self._clip_length - len(speaker_audio_clip),
                                                              speaker_emotion_clip.shape[-1]))), dim=0) \
                if self.load_emotion_s else speaker_emotion_clip
            speaker_params_clip = torch.cat((speaker_params_clip,
                                            torch.zeros(size=(self._clip_length - len(speaker_audio_clip),
                                                              speaker_params_clip.shape[-1]))), dim=0) \
                if self.load_3dmm_s else speaker_params_clip

        # ========== Load Listener Data ==========
        if self._split == "test":
            # listener's (ground-truth) face video clip
            listener_clip_length = []
            for k, listener_path in enumerate(listener_paths):
                if k != 0:
                    continue

                video_path = os.fspath(Path(self._video_dir) / listener_path.with_suffix('.mp4'))
                vr = VideoReader(video_path, ctx=cpu(0))
                clip = []
                for f in range(len(vr)):
                    frame = vr[f]
                    img = Image.fromarray(frame.asnumpy())
                    img = self._transform(img)
                    clip.append(img)
                del vr
                listener_video_clip = [torch.stack(clip, dim=0)]
            listener_video_clip = (listener_video_clip +
                                   [listener_video_clip[:1]] * (len(listener_paths) - 1))  # TODO to be modified

            listener_emotion_clip = torch.zeros(size=(0,))
            # listener's emotion ground-truths
            if self.load_emotion_l:
                listener_emotion_clips = []
                for listener_path in listener_paths:
                    emotion_path = os.fspath(Path(self._emotion_dir) / listener_path.with_suffix('.npy'))
                    emotion = np.load(emotion_path)
                    listener_clip_length.append(emotion.shape[0])
                    listener_emotion_clip = torch.from_numpy(emotion)
                    listener_emotion_clips.append(listener_emotion_clip)
                listener_emotion_clip = listener_emotion_clips

            listener_params_clip = torch.zeros(size=(0,))
            # listener's 3DMM coefficients ground-truths
            if self.load_3dmm_l:
                listener_params_clips = []
                for listener_path in listener_paths:
                    params_path = os.fspath(Path(self._3dmm_dir) / listener_path.with_suffix('.npy'))
                    params = torch.FloatTensor(np.load(params_path)).squeeze()
                    listener_params_clip = self._transform_3dmm(params)[0]
                    listener_params_clips.append(listener_params_clip)
                listener_params_clip = listener_params_clips

            speaker_clip_length = total_length
            listener_clip_length = torch.tensor(listener_clip_length)
        else:
            # listener's (ground-truth) face video clip
            listener_video_clip = torch.zeros(size=(0,))
            if self.load_video_l:
                video_path = os.fspath(Path(self._video_dir) / listener_path.with_suffix('.mp4'))
                vr = VideoReader(video_path, ctx=cpu(0))

                clip = []
                for i in range(cp, cp + self._clip_length*self.interval,self.interval):
                    if i >= len(vr):
                        break
                    frame = vr[i]
                    img = Image.fromarray(frame.asnumpy())
                    img = self._transform(img)
                    clip.append(img)
                del vr
                if len(clip)==0:
                    listener_video_clip=torch.zeros_like(speaker_video_clip)
                else:
                    listener_video_clip = torch.stack(clip, dim=0)

                #load listener past clip
                clip=[]
                vr = VideoReader(video_path, ctx=cpu(0))
                for i in range(last_cp, last_cp + self._clip_length*self.interval,self.interval):
                    if i >= len(vr):
                        break
                    frame = vr[i]
                    img = Image.fromarray(frame.asnumpy())
                    img = self._transform(img)
                    clip.append(img)
                del vr
                # listener_video_clip_last = torch.stack(clip, dim=0)
                if len(clip)==0:
                    listener_video_clip_last=torch.zeros_like(speaker_video_clip)
                else:
                    listener_video_clip_last = torch.stack(clip, dim=0)


                _clip_length = len(listener_video_clip)
                if _clip_length<self._clip_length:
                    listener_video_clip = torch.cat(
                        (listener_video_clip,
                         torch.zeros(size=(self._clip_length - _clip_length, *listener_video_clip.shape[1:]))), dim=0)
                if len(listener_video_clip_last)<self._clip_length:
                    # print('listener_video_clip_last previous shape',listener_video_clip_last.shape)
                    listener_video_clip_last = torch.cat(
                        (listener_video_clip_last,
                         torch.zeros(size=(self._clip_length - len(listener_video_clip_last), *listener_video_clip_last.shape[1:]))), dim=0)
                    # print('listener_video_clip_last after shape',listener_video_clip_last.shape)
                    # pdb.set_trace()
                    
                        
                
                
                # if self.load_video_s:
                #     listener_video_clip = torch.cat(#np.int64(total_length/self.interval)
                #         (listener_video_clip, speaker_video_clip[(_clip_length - self._clip_length):]), dim=0) \
                #         if _clip_length < self._clip_length else listener_video_clip[:self._clip_length]
                    
                #     listener_video_clip_last = torch.cat(#np.int64(total_length/self.interval)
                #         (listener_video_clip_last, speaker_video_clip[(_clip_length - self._clip_length):]), dim=0) \
                #         if _clip_length < self._clip_length else listener_video_clip[:self._clip_length]
                # else:
                #     listener_video_clip = torch.cat(
                #         (listener_video_clip,
                #          torch.zeros(size=(self._clip_length - _clip_length, *listener_video_clip.shape[1:]))), dim=0) \
                #         if _clip_length < self._clip_length else listener_video_clip[:self._clip_length]
                #     listener_video_clip_last = torch.cat(
                #     (listener_video_clip_last,
                #         torch.zeros(size=(self._clip_length - _clip_length, *listener_video_clip.shape[1:]))), dim=0) \
                #     if _clip_length < self._clip_length else listener_video_clip[:self._clip_length]

            # listener's (ground-truth) facial attribute (emotion) clip
            listener_emotion_clip = torch.zeros(size=(0,))
            if self.load_emotion_l:
                emotion_path = os.fspath(Path(self._emotion_dir) / listener_path.with_suffix('.npy'))
                emotion = np.load(emotion_path)
                assert self.load_emotion_s, "Loading speaker's emotion is required for listener's emotion at the moment"
                listener_emotion_clip = torch.from_numpy(emotion)[cp: cp + self._clip_length*self.interval:self.interval]
                _clip_length = len(listener_emotion_clip)
                listener_emotion_clip = torch.cat(
                    (listener_emotion_clip, speaker_emotion_clip[(_clip_length - self._clip_length):]), dim=0) \
                    if _clip_length < self._clip_length else listener_emotion_clip[:self._clip_length]

            # speaker's (ground-truth) 3DMM coefficient (facial motion) clip
            listener_params_clip = torch.zeros(size=(0,))
            if self.load_3dmm_l:
                params_path = os.fspath(Path(self._3dmm_dir) / listener_path.with_suffix('.npy'))
                params = torch.FloatTensor(np.load(params_path)).squeeze()
                assert self.load_3dmm_s, "Loading speaker's 3dmm is required for listener's 3dmm at the moment"
                params = params[cp: cp + self._clip_length*self.interval:self.interval]
                listener_params_clip = self._transform_3dmm(params)[0]
                _clip_length = len(listener_params_clip)
                listener_params_clip = torch.cat(
                    (listener_params_clip, speaker_params_clip[(_clip_length - self._clip_length):]), dim=0) \
                    if _clip_length < self._clip_length else listener_params_clip[:self._clip_length]

            speaker_clip_length = torch.tensor(total_length)
            listener_clip_length = torch.tensor(total_length)
        # print('listener_video_clip.shape: ',listener_video_clip.shape)
        # print('speaker_video_clip.shape',speaker_video_clip.shape)
        # print('listener_video_clip_last.shape',listener_video_clip_last.shape)
        # print('listener_emotion_clip.shape: ',listener_emotion_clip.shape)
        # print('speaker_emotion_clip.shape: ',speaker_emotion_clip.shape)
        
        # speaker_emotion_clip
        return {'img': speaker_video_clip, 'index': index,'speaker_audio_clip': speaker_audio_clip,'speaker_emotion': speaker_emotion_clip,
                                'speaker_3dmm': speaker_params_clip,'speaker_audio_path':speaker_audio_path,
                                'listener_video_clip': listener_video_clip, 'listener_align_video_clip':listener_video_clip,
                                'listener_video_clip_last':listener_video_clip_last,
                                'listener_emotion': listener_emotion_clip,
                                'listener_3dmm': listener_params_clip,'listener_audio_path':listener_audio_path,
                                'listener_reference': listener_video_clip[0],
                                # 'speaker_emotion_path':speaker_emotion_path,'listener_emotion_path':listener_emotion_path,
                                'cp':[cp,cp + self._clip_length*self.interval],'speaker_clip_length':speaker_clip_length,'listener_clip_length':listener_clip_length}
        # return (
        #     speaker_audio_clip,
        #     speaker_video_clip,
        #     speaker_emotion_clip,
        #     speaker_params_clip,
        #     listener_video_clip,
        #     listener_emotion_clip,
        #     listener_params_clip,
        #     speaker_clip_length,
        #     listener_clip_length,
        # )

    def __len__(self):
        return self._len
def get_dataset(conf, split, load_audio=True, load_video_s=False, load_video_l=False, load_emotion_s=False,
                   load_emotion_l=False, load_3dmm_s=False, load_3dmm_l=False, load_ref=False, repeat_mirrored=True,
                   parallel=False,shuffle=True,drop_last=True):
    assert split in ["train", "val", "test"], "split must be in [train, val, test]"
    # print('==> Preparing data for {}...'.format(split) + '\n')
    # dataset = ReactionDataset(conf.dataset_path,conf.real_dataset_path, split, img_size=conf.img_size, crop_size=conf.crop_size,
    #                           clip_length=conf.clip_length,
    #                           load_audio=load_audio, load_video_s=load_video_s, load_video_l=load_video_l,
    #                           load_emotion_s=load_emotion_s, load_emotion_l=load_emotion_l, load_3dmm_s=load_3dmm_s,
    #                           load_3dmm_l=load_3dmm_l, load_ref=load_ref, repeat_mirrored=repeat_mirrored)
    # pdb.set_trace()


    # # '''those are for 2024'''
    # dataset = ReactionDataset_readvideo(conf.dataset_path,conf.real_dataset_path, split, img_size=conf.img_size, crop_size=conf.crop_size,
    #                           clip_length=conf.clip_length,
    #                           load_audio=load_audio, load_video_s=load_video_s, load_video_l=load_video_l,
    #                           load_emotion_s=load_emotion_s, load_emotion_l=load_emotion_l, load_3dmm_s=load_3dmm_s,
    #                           load_3dmm_l=load_3dmm_l, load_ref=load_ref, repeat_mirrored=repeat_mirrored)
    
    # #this is for 2024 training
    # dataset = ReactionDataset_readvideo_twocrop(conf.dataset_path,conf.real_dataset_path, split, img_size=conf.img_size, crop_size=conf.crop_size,
    #                           clip_length=conf.clip_length,
    #                           load_audio=load_audio, load_video_s=load_video_s, load_video_l=load_video_l,
    #                           load_emotion_s=load_emotion_s, load_emotion_l=load_emotion_l, load_3dmm_s=load_3dmm_s,
    #                           load_3dmm_l=load_3dmm_l, load_ref=load_ref, repeat_mirrored=repeat_mirrored)

    #


    # root_dir: str = './data',  # /path/to/dataset
    #              split: str = 'train',
    #              clip_length: int = None,
    #              target_size: int = 224,
    #              crop_size: int = 224,
    #              fps: int = 30,
    #              audio_feature_type: str = 'wav2vec',
    #              load_video_s: bool = False,
    #              load_video_l: bool = False,
    #              load_audio: bool = True,
    #              load_emotion_s: bool = True,
    #              load_emotion_l: bool = True,
    #              load_3dmm_s: bool = True,
    #              load_3dmm_l: bool = True,
    #              normalize_3dmm: str = 'standard',
    '''those are for 2025'''
    dataset=ReactionDataset_2025(root_dir='/data/scratch/acw569/face_react/2025data',split=split,clip_length=conf.clip_length,crop_size=conf.crop_size,
                target_size=conf.img_size,fps=25,load_audio=load_audio, load_video_s=load_video_s, load_video_l=load_video_l,
                              load_emotion_s=load_emotion_s, load_emotion_l=load_emotion_l, load_3dmm_s=False,
                              load_3dmm_l=False,normalize_3dmm='standard')
    # dataset=ReactionDataset_2025(root_dir='/data/scratch/acw569/face_react/2025data/',split='val',clip_length=conf.clip_length,crop_size=conf.crop_size,
    #             target_size=conf.img_size,fps=25,load_audio=load_audio, load_video_s=load_video_s, load_video_l=load_video_l,
    #                           load_emotion_s=load_emotion_s, load_emotion_l=load_emotion_l, load_3dmm_s=True,
    #                           load_3dmm_l=True,normalize_3dmm='standard')
    # dataset = ReactionDataset_readvideo(conf.dataset_path,conf.real_dataset_path, split, img_size=conf.img_size, crop_size=conf.crop_size,
    #                           clip_length=conf.clip_length,
    #                           load_audio=load_audio, load_video_s=load_video_s, load_video_l=load_video_l,
    #                           load_emotion_s=load_emotion_s, load_emotion_l=load_emotion_l, load_3dmm_s=load_3dmm_s,
    #                           load_3dmm_l=load_3dmm_l, load_ref=load_ref, repeat_mirrored=repeat_mirrored)
    
    '''this is for training'''
    # dataset = ReactionDataset_readvideo_twocrop(conf.dataset_path,conf.real_dataset_path, split, img_size=conf.img_size, crop_size=conf.crop_size,
    #                           clip_length=conf.clip_length,
    #                           load_audio=load_audio, load_video_s=load_video_s, load_video_l=load_video_l,
    #                           load_emotion_s=load_emotion_s, load_emotion_l=load_emotion_l, load_3dmm_s=load_3dmm_s,
    #                           load_3dmm_l=load_3dmm_l, load_ref=load_ref, repeat_mirrored=repeat_mirrored)

    return dataset
# def get_dataloader_split(conf, dataset,split, load_audio=False, load_video_s=False, load_video_l=False, load_emotion_s=False,
#                    load_emotion_l=False, load_3dmm_s=False, load_3dmm_l=False, load_ref=False, repeat_mirrored=True,
#                    parallel=False,shuffle=True,drop_last=True):
#     if parallel and distributed.is_initialized():
#         # drop last to make sure that there is no added special indexes
#         sampler = DistributedSampler(dataset,
#                                         shuffle=shuffle,
#                                         drop_last=True)
#     else:
#         sampler = None
#     pdb.set_trace()
#     dataloader=DataLoader(
#             dataset,
#             batch_size=1,# conf.batch_size,  #EDIT
#             sampler=sampler,
#             # with sampler, use the sample instead of this option
#             shuffle=False if sampler else shuffle,
#             num_workers=conf.num_workers,
#             pin_memory=True,
#             drop_last=drop_last,
#             #multiprocessing_context=get_context('fork'),
#         )
#     # shuffle = True if split == "train" else False
#     # dataloader = DataLoader(dataset=dataset, batch_size=conf.batch_size, shuffle=shuffle, num_workers=conf.num_workers)
#     return dataloader
# def get_dataloader(conf, split, load_audio=False, load_video_s=False, load_video_l=False, load_emotion_s=False,
#                    load_emotion_l=False, load_3dmm_s=False, load_3dmm_l=False, load_ref=False, repeat_mirrored=True,
#                    parallel=False,shuffle=True,drop_last=True):
#     assert split in ["train", "val", "test"], "split must be in [train, val, test]"
#     # print('==> Preparing data for {}...'.format(split) + '\n')
#     dataset = ReactionDataset(conf.dataset_path,conf.real_dataset_path, split, img_size=conf.img_size, crop_size=conf.crop_size,
#                               clip_length=conf.clip_length,
#                               load_audio=load_audio, load_video_s=load_video_s, load_video_l=load_video_l,
#                               load_emotion_s=load_emotion_s, load_emotion_l=load_emotion_l, load_3dmm_s=load_3dmm_s,
#                               load_3dmm_l=load_3dmm_l, load_ref=load_ref, repeat_mirrored=repeat_mirrored)
#     # dataset = ReactionDataset_readvideo(conf.dataset_path,conf.real_dataset_path, split, img_size=conf.img_size, crop_size=conf.crop_size,
#     #                           clip_length=conf.clip_length,
#     #                           load_audio=load_audio, load_video_s=load_video_s, load_video_l=load_video_l,
#     #                           load_emotion_s=load_emotion_s, load_emotion_l=load_emotion_l, load_3dmm_s=load_3dmm_s,
#     #                           load_3dmm_l=load_3dmm_l, load_ref=load_ref, repeat_mirrored=repeat_mirrored)
#     if parallel and distributed.is_initialized():
#         # drop last to make sure that there is no added special indexes
#         sampler = DistributedSampler(dataset,
#                                         shuffle=shuffle,
#                                         drop_last=True)
#     else:
#         sampler = None
#     dataloader=DataLoader(
#             dataset,
#             batch_size= conf.batch_size,  #EDIT
#             sampler=sampler,
#             # with sampler, use the sample instead of this option
#             shuffle=False if sampler else shuffle,
#             num_workers=conf.num_workers,
#             pin_memory=True,
#             drop_last=drop_last,
#             #multiprocessing_context=get_context('fork'),
#         )
#     # shuffle = True if split == "train" else False
#     # dataloader = DataLoader(dataset=dataset, batch_size=conf.batch_size, shuffle=shuffle, num_workers=conf.num_workers)
#     return dataloader
