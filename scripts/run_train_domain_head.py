import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from models.dinov2_model import DinoV2Model
from models.domain_head import DomainHead
from PIL import Image
import glob
import os
from tqdm import tqdm
from core.utils import get_device

class DomainPatchDataset(Dataset):
  """
  教師付き or クラスタ擬似ラベル付きのpatch分類データセット
  data_dir の構成:
    data/train/road/*.jpg
    data/train/slope/*.jpg
  """
  def __init__(self, model, data_root, classes):
    self.samples = []
    self.model = model
    self.classes = classes
    for cls in classes:
      paths = glob.glob(f"{data_root}/{cls}/*.jpg")
      for p in paths:
        self.samples.append((p, cls))

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, idx):
    path, cls = self.samples[idx]
    img = Image.open(path).convert("RGB")
    feats = self.model.extract_features(img)
    label = torch.tensor(self.classes.index(cls))
    return feats, label

def train_domain_head():
  device = get_device()
  model = DinoV2Model(device=device)
  classes = ["road", "forest", "other"]
  dataset = DomainPatchDataset(model, "data/train", classes)
  loader = DataLoader(dataset, batch_size=1, shuffle=True)

  domain_head = DomainHead(in_dim=768, num_classes=len(classes)).to(device)
  optimizer = optim.Adam(domain_head.parameters(), lr=1e-3)
  criterion = nn.CrossEntropyLoss()

  for epoch in range(5):
    domain_head.train()
    epoch_loss = 0
    for feats, label in tqdm(loader):
      feats = feats.squeeze(0).to(device)
      feats = torch.nn.functional.normalize(feats, dim=1)
      labels = label.repeat(feats.shape[0]).to(device)
      logits = domain_head(feats)
      loss = criterion(logits, labels)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      epoch_loss += loss.item()
    print(f"[Epoch {epoch+1}] loss = {epoch_loss/len(loader):.4f}")

  os.makedirs("outputs", exist_ok=True)
  torch.save(domain_head.state_dict(), "outputs/domain_head.pth")
  print("[INFO] DomainHead trained and saved.")

if __name__ == "__main__":
  train_domain_head()
