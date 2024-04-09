from .sky_datasets import Sky
from torchvision import transforms
from .taichi_datasets import Taichi
from datasets import video_transforms
from .ucf101_datasets import UCF101
from .ffs_datasets import FaceForensics
from .ffs_image_datasets import FaceForensicsImages
from .sky_image_datasets import SkyImages
from .ucf101_image_datasets import UCF101Images
from .taichi_image_datasets import TaichiImages
# from .reaction_dataset import get_dataset
import datasets.reaction_dataset as DA_RE
import pdb
import argparse
import platform
def react_parse_arg():
    parser = argparse.ArgumentParser(description='PyTorch Training')
    # Param
    if 'sbg'  in platform.node() or 'rdg' in platform.node():
        parser.add_argument('--real-dataset-path', default="/data/scratch/acw569/face_react/", type=str, help="dataset path")
    else:
        parser.add_argument('--real-dataset-path', default="/import/nobackup_mmv_ioannisp/zs003/face_react/face_final_v2", type=str, help="dataset path")
    parser.add_argument('--dataset-path', default="./data", type=str, help="dataset path")
    parser.add_argument('--resume', default="", type=str, help="checkpoint path")
    # parser.add_argument('-b', '--batch-size', default=4, type=int, metavar='N', help='mini-batch size (default: 4)')
    parser.add_argument('-lr', '--learning-rate', default=0.00001, type=float, metavar='LR',
                        help='initial learning rate')
    parser.add_argument('-e', '--epochs', default=50, type=int, metavar='N', help='number of total epochs to run')
    # parser.add_argument('-j', '--num_workers', default=6, type=int, metavar='N',
    #                     help='number of data loading workers (default: 8)')
    parser.add_argument('--weight-decay', '-wd', default=1e-3, type=float, metavar='W',
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--optimizer-eps', default=1e-8, type=float)
    parser.add_argument('--img-size', default=256, type=int, help="size of train/test image data")
    parser.add_argument('--crop-size', default=224, type=int, help="crop size of train/test image data")
    parser.add_argument('-max-seq-len', default=751, type=int, help="max length of clip")
    parser.add_argument('--clip-length', default=64, type=int, help="len of video clip")# equal to batch size
    parser.add_argument('--window-size', default=8, type=int, help="prediction window-size for online mode")
    parser.add_argument('--feature-dim', default=128, type=int, help="feature dim of model")
    parser.add_argument('--audio-dim', default=78, type=int, help="feature dim of audio")
    parser.add_argument('--_3dmm-dim', default=58, type=int, help="feature dim of 3dmm")
    parser.add_argument('--emotion-dim', default=25, type=int, help="feature dim of emotion")
    parser.add_argument('--online', action='store_true', help='online / offline method')
    parser.add_argument('--render', action='store_true', help='w/ or w/o render')
    parser.add_argument('--momentum', type=float, default=0.99)
    parser.add_argument('--outdir', default="/import/nobackup_mmv_ioannisp/zs003/checkpoints/react/noaudio_v2", type=str, help="result dir")
    parser.add_argument('--device', default='cuda', type=str, help="device: cuda / cpu")
    parser.add_argument('--gpu-ids', type=str, default='1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--kl-p', default=0.00001, type=float, help="hyperparameter for kl-loss")
    parser.add_argument('--div-p', default=10, type=float, help="hyperparameter for div-loss")

    args = parser.parse_args()
    return args
def get_dataset(args):
    temporal_sample = video_transforms.TemporalRandomCrop(args.num_frames * args.frame_interval) # 16 1

    if args.dataset == 'ffs':
        transform_ffs = transforms.Compose([
            video_transforms.ToTensorVideo(), # TCHW
            video_transforms.RandomHorizontalFlipVideo(),
            video_transforms.UCFCenterCropVideo(args.image_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        return FaceForensics(args, transform=transform_ffs, temporal_sample=temporal_sample)
    elif args.dataset == 'ffs_img':
        transform_ffs = transforms.Compose([
            video_transforms.ToTensorVideo(), # TCHW
            video_transforms.RandomHorizontalFlipVideo(),
            video_transforms.UCFCenterCropVideo(args.image_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        return FaceForensicsImages(args, transform=transform_ffs, temporal_sample=temporal_sample)
    elif args.dataset == 'ucf101':
        transform_ucf101 = transforms.Compose([
            video_transforms.ToTensorVideo(), # TCHW
            video_transforms.RandomHorizontalFlipVideo(),
            video_transforms.UCFCenterCropVideo(args.image_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        return UCF101(args, transform=transform_ucf101, temporal_sample=temporal_sample)
    elif args.dataset == 'ucf101_img':
        transform_ucf101 = transforms.Compose([
            video_transforms.ToTensorVideo(), # TCHW
            video_transforms.RandomHorizontalFlipVideo(),
            video_transforms.UCFCenterCropVideo(args.image_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        return UCF101Images(args, transform=transform_ucf101, temporal_sample=temporal_sample)
    elif args.dataset == 'taichi':
        transform_taichi = transforms.Compose([
            video_transforms.ToTensorVideo(), # TCHW
            video_transforms.RandomHorizontalFlipVideo(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        return Taichi(args, transform=transform_taichi, temporal_sample=temporal_sample)
    elif args.dataset == 'taichi_img':
        transform_taichi = transforms.Compose([
            video_transforms.ToTensorVideo(), # TCHW
            video_transforms.RandomHorizontalFlipVideo(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        return TaichiImages(args, transform=transform_taichi, temporal_sample=temporal_sample)
    elif args.dataset == 'sky':  
        transform_sky = transforms.Compose([
                    video_transforms.ToTensorVideo(),
                    video_transforms.CenterCropResizeVideo(args.image_size),
                    # video_transforms.RandomHorizontalFlipVideo(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
            ])
        return Sky(args, transform=transform_sky, temporal_sample=temporal_sample)
    elif args.dataset == 'sky_img':  
        transform_sky = transforms.Compose([
                    video_transforms.ToTensorVideo(),
                    video_transforms.CenterCropResizeVideo(args.image_size),
                    # video_transforms.RandomHorizontalFlipVideo(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
            ])
        return SkyImages(args, transform=transform_sky, temporal_sample=temporal_sample)
    elif args.dataset == 'reaction':
        #.get_dataset
        args=react_parse_arg()
        # pdb.set_trace()
        # if kwargs['split']=='train':
        return  DA_RE.get_dataset(args, "train", load_audio=True, load_video_s=True, load_emotion_s=True, load_video_l=True,load_emotion_l=True,  load_3dmm_l=True)
        # elif kwargs['split']=='val':
        #     dataset = DA_RE.get_dataset(args, "val", load_audio=True, load_video_s=True, load_emotion_s=True, load_video_l=True,load_emotion_l=True,  load_3dmm_l=True)
        # elif kwargs['split']=='test':
        #     dataset = DA_RE.get_dataset(args, "test", load_audio=True, load_video_s=True, load_emotion_s=True, load_video_l=True,load_emotion_l=True,  load_3dmm_l=True)

    else:
        raise NotImplementedError(args.dataset)