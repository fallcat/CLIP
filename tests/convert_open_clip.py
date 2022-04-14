import torch
open_clip_checkpoint_pth = '../checkpoints/8gpu_lr5e-4_yfcc_epoch32.pt'
clip_checkpoint_pth = '../checkpoints/RN50-8gpu_lr5e-4_yfcc_epoch32-CLIP.pt'
open_clip_cp = torch.load(open_clip_checkpoint_pth, map_location=torch.device('cpu'))
open_clip_sd = {k[7:]:v for k, v in open_clip_cp['state_dict'].items()}
torch.save(open_clip_sd, clip_checkpoint_pth)

