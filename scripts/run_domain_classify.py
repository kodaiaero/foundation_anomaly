import torch
from models.dinov2_model import DinoV2Model
from models.domain_head import DomainHead
from core.domain_cluster import visualize_clusters
from PIL import Image
import numpy as np
from core.utils import get_device

def main():
  device = get_device()
  model = DinoV2Model(device=device)
  head = DomainHead(in_dim=768, num_classes=3).to(device)
  head.load_state_dict(torch.load("outputs/domain_head.pth", map_location=device))
  head.eval()

  img = Image.open("data/test/sample.jpg").convert("RGB")
  feats, hw = model.extract_features(img, return_hw=True)
  feats = torch.nn.functional.normalize(feats, dim=1)
  probs = torch.softmax(head(feats.to(device)), dim=1)
  pred = torch.argmax(probs, dim=1).cpu().numpy()

  visualize_clusters(pred, hw=hw, save_path="outputs/domain_map/sample.png")
  print("[INFO] Domain classification results saved.")

if __name__ == "__main__":
  main()
