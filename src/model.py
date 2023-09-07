import cv2
import torch
import settings
import torch.nn as nn
from torchvision import models
import matplotlib.pyplot as plt
import torch.nn.functional as F
from fvcore.nn import FlopCountAnalysis, flop_count_table
from torchvision.transforms.functional import resize, gaussian_blur


class Medigrafi(nn.Module):
  def __init__(self, pretrained=None, postprocessed=True):
    super(Medigrafi, self).__init__()
    self.transform = None
    self.features = models.densenet121().features
    self.classifier = nn.Linear(in_features=1024, out_features=14, bias=True)
    self.last_activations = None
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
    self.last_activations = features
    print("inference run")
    return out


  def create_heatmap(self, label):
    features = F.relu(self.last_activations)
    lbl_index = settings.LABELS.index(label)
    W = self.classifier.weight[lbl_index].view(1, self.classifier.weight.shape[-1], 1, 1)
    b = self.classifier.bias[lbl_index].view(1)
    cam = torch.nn.functional.conv2d(features, W, bias=b).squeeze().detach()
    cam = torch.sigmoid(cam)
    cam = cam / settings.PP_BASELINE[lbl_index]
    cam = torch.log(cam)
    cam = resize(
      cam.view(1,1,*cam.shape),
      [settings.W, settings.W],
      antialias=None
    )
    cam = gaussian_blur(cam, 21).squeeze()
    cam = cam - torch.min(cam)
    cam = cam / torch.max(cam)
    jet_colormap = plt.get_cmap('jet')
    cam = jet_colormap(cam)[:,:,:3]
    return cam

  
  def get_total_flops(self):
    image = torch.randn(1, 3, settings.W, settings.W)
    flops = FlopCountAnalysis(self, image)
    print(flop_count_table(flops))
    return flops.total()


if __name__ == "__main__":
  model = Medigrafi(pretrained="../checkpoints/densenet_weights")
  flops = model.get_total_flops()

  print("Total flops:" , flops)
