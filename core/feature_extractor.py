import torch
from PIL import Image
import glob

def compute_normal_feature_mean(model, normal_dir):
  """正常画像群から平均特徴マップを作成"""
  paths = glob.glob(f"{normal_dir}/*.jpg") + glob.glob(f"{normal_dir}/*.png")
  if not paths:
    raise ValueError(f"No images found in {normal_dir}")
  features = []
  for p in paths:
      img = Image.open(p).convert("RGB")
      feat = model.extract_features(img)
      features.append(feat)
  F_mean = torch.mean(torch.stack(features), dim=0)
  return F_mean
