from models.i3d.extract_i3d import ExtractI3D
from utils.utils import build_cfg_path
from omegaconf import OmegaConf
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.cuda.get_device_name(0)

# Select the feature type
feature_type = 'i3d'

# Load and patch the config
args = OmegaConf.load(build_cfg_path(feature_type))
args.video_paths = ['/mnt/fast/nobackup/scratch4weeks/hl01486/project/av_gen/data/datasets/vggsound/videos/0012y1s1bJI_000350.mp4']
# args.show_pred = True
args.stack_size = 24
args.step_size = 24
# args.extraction_fps = 60
args.flow_type = 'raft' # 'pwc' is not supported on Google Colab (cupy version mismatch)
# args.streams = 'flow'

# Load the model
extractor = ExtractI3D(args)

# Extract features
for video_path in args.video_paths:
    print(f'Extracting for {video_path}')
    feature_dict = extractor.extract(video_path)
    # save the key "rgb"
    [(print(k), print(v.shape), print(v)) for k, v in feature_dict.items()]