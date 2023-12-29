import os
import torch
import pickle
from omegaconf import OmegaConf
from models.clip.extract_clip import ExtractCLIP
from models.i3d.extract_i3d import ExtractI3D
from models.s3d.extract_s3d import ExtractS3D
from utils.utils import build_cfg_path
import numpy as np
from tqdm import tqdm

def extract_features(extractor, video_path, feature_key, save_folder):
    save_path = os.path.join(save_folder, f'{video_name}_{feature_key}.pkl')
    
    if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
        return
    else:
        open(save_path, 'w').close()

    feature_dict = extractor.extract(video_path)
    feature = feature_dict.get(feature_key)

    if feature is not None:
        video_name = os.path.basename(video_path).split('.')[0]
        with open(save_path, 'wb') as f:
            pickle.dump(feature, f)

def setup_extractor_and_extract_features(video_folder, feature_type, model_name, feature_key, save_folder):
    # Load and patch the config
    args = OmegaConf.load(build_cfg_path(feature_type))
    args.feature_type = feature_type
    args.model_name = model_name
    args.video_paths = [os.path.join(video_folder, f) for f in os.listdir(video_folder) if f.endswith('.mp4')]
    np.random.shuffle(args.video_paths)

    # Load the model based on type
    if feature_type == 'clip':
        args.batch_size = 32
        args.extraction_fps = 12.5
        extractor = ExtractCLIP(args)
    elif feature_type == 'i3d':
        args.stack_size = 24
        args.step_size = 24
        args.flow_type = 'raft'
        extractor = ExtractI3D(args)
    elif feature_type == 's3d':
        args.stack_size = 24
        args.step_size = 24
        extractor = ExtractS3D(args)
    else:
        raise ValueError("Invalid feature type")

    # Extract features for each video
    for video_path in tqdm(args.video_paths):
        extract_features(extractor, video_path, feature_key, save_folder)

# Set up paths and parameters
video_folder = '/mnt/fast/nobackup/scratch4weeks/hl01486/project/av_gen/data/datasets/vggsound/videos'
clip_features_folder = '/mnt/fast/nobackup/scratch4weeks/hl01486/project/av_gen/data/datasets/vggsound/video_feature_msc/clip'  # Replace with your path
i3d_features_folder = '/mnt/fast/nobackup/scratch4weeks/hl01486/project/av_gen/data/datasets/vggsound/video_feature_msc/i3d'    # Replace with your path
s3d_features_folder = '/mnt/fast/nobackup/scratch4weeks/hl01486/project/av_gen/data/datasets/vggsound/video_feature_msc/s3d'    # Replace with your path

# Extract features using CLIP, I3D, and S3D models
setup_extractor_and_extract_features(video_folder, 'clip', 'ViT-B/32', 'clip', clip_features_folder)
setup_extractor_and_extract_features(video_folder, 'i3d', None, 'rgb', i3d_features_folder)
setup_extractor_and_extract_features(video_folder, 's3d', None, 's3d', s3d_features_folder)
