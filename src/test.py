import os
import torch
import settings
import numpy as np
from tqdm import tqdm
from model import Medigrafi
from dataset import XRayDataset
from utils import load_test_transforms
from sklearn.metrics import roc_auc_score


if __name__ == "__main__":
  dataloader = torch.utils.data.DataLoader(
    XRayDataset(
      image_path="../data/images/",
      csv_path="../data/nih_labels.csv",
      split="test",
      transform = load_test_transforms()
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
