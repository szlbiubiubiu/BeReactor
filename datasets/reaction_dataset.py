"""Paired speaker/listener clips for reaction training."""

import os
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from decord import VideoReader, cpu
from PIL import Image
from torch.utils import data
from torchvision import transforms

class Transform(object):
    def __init__(self, img_size=256, crop_size=224):#align with unet 256 resolution
        self.img_size = img_size
        self.crop_size = crop_size

    def __call__(self, img):
        transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5])
        ])
        img = transform(img)
        return img

def extract_audio_features(audio_path, fps, n_frames):
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

        curr_feats.append(curr_feat)

    curr_feats = np.stack(curr_feats, axis=0)
    return curr_feats

class ReactionDataset_2025_readbothcrop(data.Dataset):
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
        self._video_dir = os.path.join(dataset_dir, 'video-face-crop')#original crop
        self.crop_video_dir = os.path.join(dataset_dir, 'ZL_crop_test')     #face detection crop, retinaface
        self._emotion_dir = os.path.join(dataset_dir, 'facial-attributes')
        self._3dmm_dir = os.path.join(dataset_dir, 'coefficients')

        speaker_path_list = []
        listener_path_list = []
        print(self._video_dir)
        for root, _, files in os.walk(self._video_dir):
            for path in files:
                path = Path(path)
                file, ext = path.stem, path.suffix
                if ext.lower() != '.mp4':
                    continue

                parts = Path(root).parts
                file_path = Path(*parts[-2:]) / file
                role = parts[-2]
                session_id = Path(parts[-1])
                gt_session_id = 'speaker' / session_id if role == 'listener' else 'listener' / session_id
                listener_file_path = gt_session_id / file

                speaker_path_list.append(file_path)
                listener_path_list.append(listener_file_path)

        self.speaker_path_list = speaker_path_list.copy()
        self.listener_path_list = listener_path_list.copy()

        if self.load_3dmm_s or self.load_3dmm_l:
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

        speaker_video_path = os.fspath(Path(self._video_dir) / speaker_path.with_suffix('.mp4'))
        vr = VideoReader(speaker_video_path, ctx=cpu(0))
        vr_crop = VideoReader(os.fspath(Path(self.crop_video_dir) / speaker_path.with_suffix('.mp4')), ctx=cpu(0))
        total_length = len(vr)  # length from 58 to more than 20000
        if not self.load_video_s:
            del vr

        cp = random.randint(0, total_length - 1 - self._clip_length*self.interval) if (self._clip_length*self.interval) < total_length else 0
        last_cp= cp if (cp<=self._clip_length*self.interval) else cp-self._clip_length*self.interval
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
            if len(speaker_video_clip)<self._clip_length:

                speaker_video_clip = torch.cat((speaker_video_clip,
                                                torch.zeros(size=(self._clip_length - len(speaker_video_clip),
                                                                  *speaker_video_clip.shape[1:]))), dim=0)
            clip = []
            for i in range(cp,cp + self._clip_length*self.interval,self.interval):
                if i >= len(vr_crop):
                    break
                frame = vr_crop[i]
                img = Image.fromarray(frame.asnumpy())
                img = self._transform(img)
                clip.append(img)
            del vr_crop
            if len(clip) == 0:
                speaker_video_clip_crop = torch.zeros_like(speaker_video_clip)
            else:
                speaker_video_clip_crop = torch.stack(clip, dim=0)
            if len(speaker_video_clip_crop)<self._clip_length:

                speaker_video_clip_crop = torch.cat((speaker_video_clip_crop,
                                                torch.zeros(size=(self._clip_length - len(speaker_video_clip_crop),
                                                                  *speaker_video_clip_crop.shape[1:]))), dim=0)

        speaker_emotion_clip = torch.zeros(size=(0,))
        if self.load_emotion_s:
            emotion_path = os.fspath(Path(self._emotion_dir) / speaker_path.with_suffix('.npy'))
            emotion = np.load(emotion_path)
            speaker_emotion_clip = torch.from_numpy(emotion)
            speaker_emotion_clip = speaker_emotion_clip[cp: cp + self._clip_length*self.interval:self.interval]  # (25-d)
            if len(speaker_emotion_clip)<self._clip_length:
                speaker_emotion_clip=torch.cat((speaker_emotion_clip,torch.zeros(self._clip_length-len(speaker_emotion_clip),speaker_emotion_clip.shape[-1])),dim=0)
        speaker_params_clip = torch.zeros(size=(0,))
        if self.load_3dmm_s:
            params_path = os.fspath(Path(self._3dmm_dir) / speaker_path.with_suffix('.npy'))
            params = torch.FloatTensor(np.load(params_path)).squeeze()
            params = params[cp: cp + self._clip_length*self.interval:self.interval]
            speaker_params_clip = self._transform_3dmm(params)[0]  # (58-d)

        speaker_audio_clip = torch.zeros(size=(0,))
        if self.load_audio:
            if self.audio_feature_type == 'wav2vec':
                audio_path = os.fspath(Path(self._audio_dir) / speaker_path.with_suffix('.npy'))
                speaker_audio_clip = np.load(audio_path)  # (768-d)
                listener_audio_path=os.fspath(Path(self._audio_dir) / listener_path.with_suffix('.npy'))
            elif self.audio_feature_type == 'mfcc':
                audio_path = os.fspath(Path(self._audio_dir) / speaker_path.with_suffix('.wav'))
                speaker_audio_clip = extract_audio_features(audio_path, self._fps, total_length)  # (78-d)
                listener_audio_path=os.fspath(Path(self._audio_dir) / listener_path.with_suffix('.wav'))
            else:
                raise ValueError(f"Unknown audio feature type: {self.audio_feature_type}")
            speaker_audio_clip = torch.from_numpy(speaker_audio_clip)[cp: cp + self._clip_length*self.interval:self.interval]
            speaker_audio_path=audio_path
            listener_audio_path=listener_audio_path

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

        listener_video_clip = torch.zeros(size=(0,))
        if self.load_video_l:
            listener_video_path = os.fspath(Path(self._video_dir) / listener_path.with_suffix('.mp4'))
            vr = VideoReader(listener_video_path, ctx=cpu(0))
            vr_crop= VideoReader(os.fspath(Path(self.crop_video_dir) / listener_path.with_suffix('.mp4')), ctx=cpu(0))
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
            listener_video_path = os.fspath(Path(self._video_dir) / listener_path.with_suffix('.mp4'))
            clip = []
            for i in range(cp, cp + self._clip_length*self.interval,self.interval):
                if i >= len(vr_crop):
                    break
                frame = vr_crop[i]
                img = Image.fromarray(frame.asnumpy())
                img = self._transform(img)
                clip.append(img)
            del vr_crop
            if len(clip)==0:
                listener_video_clip_crop=torch.zeros_like(speaker_video_clip_crop)
            else:
                listener_video_clip_crop = torch.stack(clip, dim=0)

            clip=[]
            vr = VideoReader(listener_video_path, ctx=cpu(0))
            for i in range(last_cp, last_cp + self._clip_length*self.interval,self.interval):
                if i >= len(vr):
                    break
                frame = vr[i]
                img = Image.fromarray(frame.asnumpy())
                img = self._transform(img)
                clip.append(img)
            del vr
            if len(clip)==0:
                listener_video_clip_last=torch.zeros_like(speaker_video_clip)
            else:
                listener_video_clip_last = torch.stack(clip, dim=0)

            _clip_length = len(listener_video_clip)
            if _clip_length<self._clip_length:
                listener_video_clip = torch.cat(
                    (listener_video_clip,
                     torch.zeros(size=(self._clip_length - _clip_length, *listener_video_clip.shape[1:]))), dim=0)
            if len(listener_video_clip_crop)<self._clip_length:
                listener_video_clip_crop = torch.cat(
                    (listener_video_clip_crop,
                     torch.zeros(size=(self._clip_length - len(listener_video_clip_crop), *listener_video_clip_crop.shape[1:]))), dim=0)
            if len(listener_video_clip_last)<self._clip_length:
                listener_video_clip_last = torch.cat(
                    (listener_video_clip_last,
                     torch.zeros(size=(self._clip_length - len(listener_video_clip_last), *listener_video_clip_last.shape[1:]))), dim=0)

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

        return {'img': speaker_video_clip, 'index': index,'speaker_audio_clip': speaker_audio_clip,
                'speaker_video_clip_crop':speaker_video_clip_crop, 'speaker_emotion': speaker_emotion_clip,
                                'speaker_3dmm': speaker_params_clip,
                                'speaker_video_path':speaker_video_path,'speaker_audio_path':speaker_audio_path,
                                'listener_video_path':listener_video_path,
                                'listener_video_clip': listener_video_clip, 'listener_align_video_clip':listener_video_clip,
                                'listener_video_clip_crop':listener_video_clip_crop,
                                'listener_video_clip_last':listener_video_clip_last,
                                'listener_emotion': listener_emotion_clip,
                                'listener_3dmm': listener_params_clip,'listener_audio_path':listener_audio_path,
                                'listener_reference': listener_video_clip[0],
                                'cp':[cp,cp + self._clip_length*self.interval],'speaker_clip_length':speaker_clip_length,'listener_clip_length':listener_clip_length}

    def __len__(self):
        return self._len
