#!/usr/bin/env python3
# Masked Autoencoder Video Reconstruction Visualization

import os
import torch
import numpy as np
import cv2
from slowfast.models.masked import MaskMViT
from slowfast.utils.parser import load_config
from slowfast.utils.checkpoint import load_checkpoint
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

# --- CONFIG ---
VIDEO_PATH = "/content/drive/MyDrive/Mae/data/part_0/--q_mvQ8zP8_000067_000077.mp4"
CKPT_PATH = "/content/drive/MyDrive/Mae/data/VIT_B_16x4_MAE_PT_FUSED.pyth"  # Use ORIGINAL
CFG_PATH = "/content/drive/MyDrive/Mae/SlowFast/configs/masked_ssl/k400_VIT_B_16x4_MAE_PT.yaml"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MASK_RATIOS = [0.75, 0.9, 0.95]
OUTPUT_DIR = "/content/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- LOAD CONFIG ---
import argparse
args = argparse.Namespace(opts=None, num_shards=1, shard_id=0)
cfg = load_config(args, CFG_PATH)
cfg.NUM_GPUS = 1
cfg.TEST.CHECKPOINT_FILE_PATH = CKPT_PATH

# IMPORTANT: Ensure SEPARATE_QKV matches your checkpoint
# Your original checkpoint has separate Q/K/V, so this should be True
if not hasattr(cfg.MVIT, 'SEPARATE_QKV'):
    cfg.MVIT.SEPARATE_QKV = True
print(f"Config SEPARATE_QKV: {cfg.MVIT.SEPARATE_QKV}")
print(f"Config TIME_STRIDE_LOSS: {cfg.MASK.TIME_STRIDE_LOSS}")

# --- LOAD MODEL ---
model = MaskMViT(cfg)
model = model.to(DEVICE)
model.eval()
load_checkpoint(CKPT_PATH, model, None, inflation=False, convert_from_caffe2=False)

# --- Define Normalization Stats ---
norm_mean = [0.45, 0.45, 0.45]  # From checkpoint config
norm_std = [0.225, 0.225, 0.225]

# --- LOAD VIDEO FRAMES ---
def load_video_frames(video_path, num_frames, crop_size):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = torch.linspace(0, total_frames - 1, num_frames).long().tolist()
    
    frames = []
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    
    if len(frames) == 0:
        raise ValueError("No frames could be loaded from the video")

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((crop_size, crop_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std)
    ])
    
    processed_frames = [transform(frame) for frame in frames]
    video_tensor = torch.stack(processed_frames, dim=1).unsqueeze(0)
    return video_tensor

# --- De-normalize for visualization ---
def de_normalize(tensor):
    mean = torch.tensor(norm_mean, device=tensor.device)
    std = torch.tensor(norm_std, device=tensor.device)
    tensor = tensor * std + mean
    return (tensor.clamp(0, 1) * 255).byte().cpu().numpy()

# --- VISUALIZATION ---
def visualize(original, masked, reconstructed, mask_ratio, output_filename):
    T = original.shape[0]
    fig, axes = plt.subplots(T, 3, figsize=(9, 3 * T))
    fig.suptitle(f"Mask Ratio: {mask_ratio*100:.0f}%", fontsize=16)
    
    for i in range(T):
        title_original = "Original" if i == 0 else ""
        title_masked = "Masked" if i == 0 else ""
        title_reconstructed = "Reconstructed" if i == 0 else ""

        axes[i, 0].imshow(original[i])
        axes[i, 0].set_title(title_original)
        axes[i, 1].imshow(masked[i])
        axes[i, 1].set_title(title_masked)
        axes[i, 2].imshow(reconstructed[i])
        axes[i, 2].set_title(title_reconstructed)
        
        for j in range(3):
            axes[i, j].axis('off')
            
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_filename)
    plt.close(fig)
    print(f"Saved visualization to {output_filename}")

if __name__ == "__main__":
    frames = load_video_frames(VIDEO_PATH, cfg.DATA.NUM_FRAMES, cfg.DATA.TEST_CROP_SIZE)
    frames = frames.to(DEVICE)

    for mask_ratio in MASK_RATIOS:
        with torch.no_grad():
            latent, mask, ids_restore, thw = model._mae_forward_encoder(frames, mask_ratio=mask_ratio)
            pred = model._mae_forward_decoder(latent, ids_restore, mask, thw)
            
            # Handle decoder output format
            if isinstance(pred, (list, tuple)):
                try:
                    pred = next(p['pred'] for p in reversed(pred) if isinstance(p, dict) and 'pred' in p)
                except StopIteration:
                    pred = pred[0] if len(pred) > 0 else pred
            if isinstance(pred, dict) and 'pred' in pred:
                pred = pred['pred']
            
            # TIME_STRIDE_LOSS=True: subsample frames for visualization
            if cfg.MASK.TIME_STRIDE_LOSS:
                im_viz = frames[:, :, ::cfg.MVIT.PATCH_STRIDE[0], :, :]
            else:
                im_viz = frames
            
            # Patchify with TIME_STRIDE_LOSS setting
            patches = model._patchify(im_viz, p=16, time_stride_loss=cfg.MASK.TIME_STRIDE_LOSS)
            
            print(f"\nMask ratio {mask_ratio}:")
            print(f"  pred shape: {pred.shape}")
            print(f"  patches shape: {patches.shape}")
            print(f"  mask shape: {mask.shape}")
            
            # Get batch size
            N = frames.shape[0]
            
            # Initialize reconstruction with original patches
            reconstruction_patches = patches.clone()
            
            # Create mask indices - mask is 1 for masked patches, 0 for visible
            mask_indices = mask.squeeze(-1).bool()  # [N, num_patches]
            
            # Replace only the masked patches with predictions
            reconstruction_patches[mask_indices] = pred.reshape(-1, pred.shape[-1])
            
            # Reshape mask for masked video
            mask_reshaped = mask.reshape(N, -1, 1)
            
            # Unpatchify to get videos
            reconstructed_video = model._unpatchify(reconstruction_patches)
            masked_video = model._unpatchify(patches * (1 - mask_reshaped))
            
            # Prepare tensors for visualization
            original_viz = im_viz.squeeze(0).permute(1, 2, 3, 0)
            masked_viz = masked_video.squeeze(0).permute(1, 2, 3, 0)
            reconstructed_viz = reconstructed_video.squeeze(0).permute(1, 2, 3, 0)
            
            original_viz = de_normalize(original_viz)
            masked_viz = de_normalize(masked_viz)
            reconstructed_viz = de_normalize(reconstructed_viz)
            
            output_filename = os.path.join(OUTPUT_DIR, f"reconstruction_mask_ratio_{int(mask_ratio*100)}.png")
            visualize(original_viz, masked_viz, reconstructed_viz, mask_ratio, output_filename)
