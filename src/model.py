import torch
import settings
import torch.nn as nn
from tabulate import tabulate
from torchvision import models
import torch.nn.functional as F


class Medigrafi(nn.Module):
  def __init__(self, pretrained=None, postprocessed=False):
    super(Medigrafi, self).__init__()
    self.features = models.densenet121().features
    self.classifier = nn.Linear(in_features=1024, out_features=14, bias=True)
    if pretrained is not None:
      self.load_weights(pretrained)
    self.pp = postprocessed
    if self.pp:
      self.alpha = torch.tensor(settings.PP_WEIGHTS) / torch.tensor(settings.PP_BASELINE)


  def load_weights(self, path):
    with open(path, 'rb') as f:
      state = torch.load(f)
    self.features.load_state_dict(state['features'])
    self.classifier.load_state_dict(state['classifier'])
    print("+ Weights are loaded successfully from: {}".format(path))

  def forward(self, x):
    features = self.features(x)
    out = F.relu(features, inplace=True)
    out = F.adaptive_avg_pool2d(out, (1, 1))
    out = torch.flatten(out, 1)
    out = torch.sigmoid(self.classifier(out))
    if self.pp:
      f1 = 1 - torch.exp(-self.alpha * out / (1 - out))
      f2 = 1 + torch.exp(-self.alpha * out / (1 - out))
      out = f1 / f2
    return [features, out]

  def report(self, o):
    for i in range(o.shape[0]):
      print(f"== Sample {i + 1} ==")
      print(
        tabulate(
          [[lbl, round(score.item(), 3)] for lbl, score in zip(settings.LABELS, o[i])],
          headers=["Disease", "Score"],
        )
      )
      print()

  def export_weights_for_heatmap(self):
    weight = self.classifier.weight
    bias = self.classifier.bias
    return weight, bias

