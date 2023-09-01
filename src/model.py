import torch
import torch.nn as nn
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
    self.pp_weights = {
      "ps": torch.tensor([
        0.61, 0.61, 0.58, 0.58, 0.31, 0.5, 0.68,
        0.62, 0.9, 0.63, 0.41, 0.63, 0.77, 0.45
      ]),
      "baseline": torch.tensor([
        0.103, 0.025, 0.119, 0.177, 0.051, 0.056, 0.012,
        0.047, 0.042, 0.021, 0.022, 0.015, 0.030, 0.002
      ])
    }

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
      f1 = 1 - torch.exp(-(self.pp_weights['ps'] / self.pp_weights['baseline']) * (out / (1 - out)))
      f2 = 1 + torch.exp(-(self.pp_weights['ps'] / self.pp_weights['baseline']) * (out / (1 - out)))
      out = f1 / f2
    return [features, out]

if __name__ == "__main__":
  from PIL import Image
  from torchvision import transforms

  model = Medigrafi(pretrained="../checkpoints/densenet_weights", postprocessed=True)
  model.eval()
  
  image = Image.open("../cardiomegaly.png")
  image = image.convert("RGB")
  tra = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
      mean=(0.485, 0.456, 0.406),
      std=(0.229, 0.224, 0.225)
    )
  ])

  image = tra(image).unsqueeze(0)
  f, o = model(image)
  print(o)
