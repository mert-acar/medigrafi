import os
from PIL import Image 
from pandas import read_csv
from torch.utils.data import Dataset


class XRayDataset(Dataset):
  def __init__(self, images_path, index_path, split=None, transforms=None, random_state=0):
    self.image_path = image_path
    self.image_index = read_csv(index_path)
    if split is not None:
      self.image_index = self.image_index[self.image_index['Fold'] == split]
    self.transforms = transforms
    self.labels = [
      "Atelectasis",
      "Cardiomegaly",
      "Effusion",
      "Infiltration",
      "Mass",
      "Nodule",
      "Pneumonia",
      "Pneumothorax",
      "Consolidation",
      "Edema",
      "Emphysema",
      "Fibrosis",
      "Pleural_Thickening",
      "Hernia"
    ]

  def __len__(self):
    return len(self.image_index)

  def __getitem__(self, idx):
    image = Image.open(os.path.join(self.image_path, self.image_index["Image Index"].iloc[idx])) 
    image = image.convert("RGB")
    labels = self.image_index.iloc[idx][self.labels].to_list()
    if self.transforms:
      image = self.transforms(image)
    return image, labels


if __name__ == "__main__":
  from torch.utils.data import DataLoader
  from torchvision import transforms as T

  csv_path = "../data/nih_labels.csv"
  image_path = "../data/images/"
  transform = T.Compose([
    T.Resize(224),
    T.ToTensor(),
    T.Normalize(
      mean=(0.485, 0.456, 0.406),
      std=(0.229, 0.224, 0.225)
    )
  ])

  dataloader = DataLoader(
    XRayDataset(
      image_path,
      csv_path,
      split="test",
      transforms=transform
    ),
    batch_size=2,
    shuffle=True
  )

  sampleX, sampleY = next(iter(dataloader))
  print("Image Shape:", sampleX.shape)
  print("Label Shape:", sampleY.shape)
