from models.clip.extract_clip import ExtractCLIP
from utils.utils import build_cfg_path
from omegaconf import OmegaConf
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.cuda.get_device_name(0)

# Select the feature type 
# ('CLIP-ViT-B-32', 'CLIP-ViT-B-16', 'CLIP-RN50x16', 'CLIP-RN50x4','CLIP-RN101', 'CLIP-RN50', 'CLIP-custom')
feature_type = 'clip'
model_name = 'ViT-B/32'

# Load and patch the config
args = OmegaConf.load(build_cfg_path(feature_type))
args.feature_type = feature_type
args.video_paths = ['/mnt/fast/nobackup/scratch4weeks/hl01486/project/av_gen/data/datasets/vggsound/videos/0012y1s1bJI_000350.mp4']
args.batch_size = 32
args.extraction_fps = 12.5
# args.show_pred = True
# args.pred_texts = ['a dog smiles', 'a woman is lifting']  # if None, does zero-shot on Kinetics 400 classes

# Load the model
extractor = ExtractCLIP(args)

# Extract features
for video_path in args.video_paths:
    print(f'Extracting for {video_path}')
    feature_dict = extractor.extract(video_path)
    # save the key "clip"
    [(print(k), print(v.shape), print(v)) for k, v in feature_dict.items()]