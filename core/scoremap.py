import torch
import numpy as np
import matplotlib.pyplot as plt
import os

def compute_scoremap(F_test, F_mean):
  score = torch.norm(F_test - F_mean, dim=1)  # [N_patches]
  size = int(np.sqrt(score.shape[0]))
  score_map = score.reshape(size, size)
  # ゼロ割対策
  mn, mx = score_map.min(), score_map.max()
  denom = (mx - mn).clamp(min=1e-8)
  score_map = (score_map - mn) / denom
  return score_map

def save_scoremap(score_map, out_path):
    """スコアマップを画像として保存"""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(4,4))
    plt.imshow(score_map, cmap="hot")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()
