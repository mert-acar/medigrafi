import cv2
import torch
import settings
import numpy as np
import torch.nn as nn
from tabulate import tabulate
from torchvision import models
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms as T
from fvcore.nn import FlopCountAnalysis, flop_count_table
from torchvision.transforms.functional import resize, gaussian_blur


class Medigrafi(nn.Module):
  def __init__(self, pretrained=None, postprocessed=False):
    super(Medigrafi, self).__init__()
    self.transform = None
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


  def create_heatmap(self, image, label, features):
    lbl_index = settings.LABELS.index(label)
    features = torch.clip(features, min=0, max=None)
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
    cam = gaussian_blur(cam, 21).squeeze().numpy()
    image = image.permute(1,2,0).numpy() # [224, 224, 3]
    image = settings.IMAGENET_STD * image + settings.IMAGENET_MEAN
    image = np.clip(image, 0, 1).astype(np.float32)
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)
    jet_colormap = plt.get_cmap('jet')
    cam = jet_colormap(cam).astype(np.float32)[:,:,:3]
    cam = cam[:, :, ::-1]
    heatmap = cv2.addWeighted(image, 0.5, cam, 0.5, 0)
    return heatmap

  
  def load_test_transforms(self):
    self.transform = T.Compose([
      T.ToTensor(),
      T.Resize(settings.W, antialias=None),
      T.Normalize(
        mean=settings.IMAGENET_MEAN,
        std=settings.IMAGENET_STD
      )
    ])
    self.eval()


  def get_total_flops(self):
    image = torch.randn(1, 3, settings.W, settings.W)
    flops = FlopCountAnalysis(self, image)
    print(flop_count_table(flops))
    return flops.total()


  def predict(self, image, heatmap=None):
    """
      Parameters
      ==========
      + image:
        Supply the RGB image as an argument to run prediction on it. Example:
        '''
          from utils import imread
          image = imread("/path/to/image/or/dicom")
        '''
      + heatmap:
        The heatmap for the selected disease. If no disease is specified the
        heatmap is created for the most probable disease above the positve treshold.
        Examples:
        '''
          # Create the heatmap for the positive diagnoses that scored the highest.
          probs, heatmap = model.predict(image, heatmap=True) 

          # Create the heatmap for a specific disease.
          probs, heatmap = model.predict(image, heatmap="Cardiomegaly")

          # Run the inference with no heatmap calculation.
          probs = model.predict(image)

      Returns
      =======
      The output probabilities and optionally the heatmap for given disease

    """

    if self.transform is None:
      self.load_test_transforms() 

    with torch.no_grad():
      image = self.transform(image).unsqueeze(0)
      features, probs = self.forward(image)
      preds = [[lbl, round(prob.item(), 4)] for (lbl, prob) in zip(settings.LABELS, probs[0])]
      pos_preds = list(
        sorted(
          filter(
            lambda x: x[1] > settings.PREDICTION_THRESHOLD,
            preds
          ),
          key=lambda x: x[1],
          reverse=True
        )
      )
    if heatmap is not None and heatmap is not False:
      if isinstance(heatmap, str) and heatmap in [x[0] for x in pos_preds]:
        label = heatmap
      elif isinstance(heatmap, bool) and heatmap==True and len(pos_preds) > 0:
        label = pos_preds[0][0]
      else:
        self.report(probs)
        raise RuntimeError("Error with positive diagnosis! The label could not be found")
      heatmap = self.create_heatmap(image[0], label, features[0])
      return pos_preds, heatmap
    return pos_preds


if __name__ == "__main__":
  model = Medigrafi(
    pretrained="../checkpoints/densenet_weights",
    postprocessed=True
  )
  flops = model.get_total_flops()

  print("Total flops:" , flops)
