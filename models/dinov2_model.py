import torch
from transformers import AutoModel, AutoImageProcessor
from .base_model import BaseFeatureExtractor

class DinoV2Model(BaseFeatureExtractor):
  def __init__(self, model_name="facebook/dinov2-base", device=None):
    self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[INFO] Using device: {self.device}")
    self.processor = AutoImageProcessor.from_pretrained(model_name)
    self.model = AutoModel.from_pretrained(model_name).to(self.device)
    self.model.eval()

  def extract_features(self, img):
    inputs = self.processor(images=img, return_tensors="pt").to(self.device)
    with torch.no_grad():
        outputs = self.model(**inputs)
    # last_hidden_state: [1, 1+N_patches, dim] 先頭のCLSを除外
    feats = outputs.last_hidden_state.squeeze(0)[1:].cpu()  # [N_patches, dim]
    return feats

