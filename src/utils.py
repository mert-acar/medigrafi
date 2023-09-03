import cv2
import torch
import pydicom
import settings
import numpy as np
import matplotlib.pyplot as plt
from pydicom.pixel_data_handlers.util import apply_voi_lut
from torchvision.transforms.functional import resize, gaussian_blur

def read_dicom(path, model_ready=False):
  dcm_file = pydicom.dcmread(path)
  # Apply default dicom calibration using the parameters inside the file
  image = apply_voi_lut(dcm_file.pixel_array, dcm_file)

  # Invert if monochrome1
  if dcm_file.PhotometricInterpretation == "MONOCHROME1":
    image = np.amax(image) - image

  # Normalization
  image = image - np.min(image)
  image = image / np.max(image)
  image = (image * 255).astype(np.uint8)
  if model_ready:
    # Dicom files are usually monochromatic, resulting in 
    # W x H images. Since the model accepts RGB images we 
    # repeat the image on a new axis to create [W x H x 3]
    image = np.stack([image] * 3, axis=-1)
  return image


def create_heatmap(image, label, features, weight, bias):
  lbl_index = settings.LABELS.index(label)
  features = torch.clip(features, min=0, max=None)
  W = weight[lbl_index].view(1, weight.shape[-1], 1, 1)
  b = bias[lbl_index].view(1)
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
  

if __name__ == "__main__":
  path = "../data/sample.dcm"
  image = read_dicom(path)
  print(f"Image Shape: {image.shape}\nImage Max Pixel: {np.max(image)}\nImage Min Pixel: {np.min(image)}")
  cv2.imshow(path, image)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
