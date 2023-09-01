import os
import torch
import numpy as np
from tqdm import tqdm
from model import Medigrafi
from dataset import XRayDataset
from torchvision import transforms as T
from sklearn.metrics import roc_auc_score


if __name__ == "__main__":
  transform = T.Compose([
    T.Resize(224),
    T.ToTensor(),
    T.Normalize(
      mean=(0.485, 0.456, 0.406),
      std=(0.229, 0.224, 0.225)
    )
  ])

  dataloader = torch.utils.data.DataLoader(
    XRayDataset(
      image_path="../data/images/",
      csv_path="../data/nih_labels.csv",
      split="test",
      transforms=transform
    ),
    batch_size=4,
    num_workers=4,
    shuffle=True
  )

  model = Medigrafi(
    pretrained="../checkpoints/densenet_weights",
    posprocessed=True
  )
  model.eval()

  auc_total = np.zeros((1,14))
  for img, labels in tqdm(dataloader):
    _, preds = model(img)
    aucs = roc_auc_score(labels, preds, average=None)
    auc_total += aucs 

  auc_total /= len(dataloader)
  for label, auc in zip(dataloader.dataset.labels, auc_total):
    print(label + ":", auc)
