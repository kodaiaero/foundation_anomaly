import torch
from transformers import AutoModel, AutoImageProcessor
from .base_model import BaseFeatureExtractor
from core.utils import get_device

class DinoV2Model(BaseFeatureExtractor):
  def __init__(self, model_name="facebook/dinov2-base", device=None):
    if device is None:
      device = get_device()

    self.device = device
    print(f"[INFO] Using device: {self.device}")

    self.processor = AutoImageProcessor.from_pretrained(model_name)
    self.model = AutoModel.from_pretrained(model_name).to(self.device)
    self.model.eval()

  def extract_features(self, img, return_hw=False):
    inputs = self.processor(images=img, return_tensors="pt").to(self.device)
    with torch.no_grad():
      outputs = self.model(**inputs)
    # last_hidden_state: [1, 1+N_patches, dim] 先頭のCLSを除外
    toks = outputs.last_hidden_state.squeeze(0)
    feats = toks[1:].cpu()
    if return_hw:
      N = feats.shape=[0]
      S = int(N ** 0.5)
      return feats, (S, S)
    return feats
