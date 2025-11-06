import torch.nn as nn
import torch.nn.functional as F

class DomainHead(nn.Module):
  """
  DINOv2などの特徴を入力として
  patchごとのドメイン分類を行う軽量なMLPヘッド
  """
  def __init__(self, in_dim=768, num_classes=3, hidden_dim=256):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(in_dim, hidden_dim),
      nn.ReLU(),
      nn.Linear(hidden_dim, num_classes)
    )
  
  def forward(self, x):
    return self.net(x)
  
  def predict_proba(self, x):
    return F.softmax(self.forward(x), dim=1)
  