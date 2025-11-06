import torch
from sklearn.cluster import KMeans
import numpy as np
from PIL import Image
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt

def extract_patch_features(model, data_dir, max_images=200):
  """指定ディレクトリの画像群からpatch特徴を抽出"""
  paths = glob.glob(f"{data_dir}/*.jpg") + glob.glob(f"{data_dir}/*.png")
  all_feats = []
  for p in tqdm(paths[:max_images]):
    img = Image.open(p).convert("RGB")
    feats = model.extract_features(img)
    all_feats.append(feats)
  return torch.cat(all_feats, dim=0)

def cluster_domains(model, data_dir, n_clusters=5, max_images=200):
  """教師なしクラスタリングでドメインを自動分割"""
  feats = extract_patch_features(model, data_dir, max_images)
  feats_np = feats.numpy()
  kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(feats_np)
  return kmeans

def visualize_clusters(cluster_ids, hw, save_path):
  """patchごとのクラスタIDを可視化"""
  H, W = hw
  cmap = plt.get_cmap("tab10")
  plt.imshow(cluster_ids.reshape(H, W), cmap=cmap)
  plt.axis("off")
  plt.savefig(save_path, bbox_inches="tight")
  plt.close()
