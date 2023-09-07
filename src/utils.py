import cv2
import pydicom
import settings
import numpy as np
from PIL import Image
from time import time
from tqdm import tqdm
from torchvision import transforms as T 
from pydicom.errors import InvalidDicomError
from pydicom.pixel_data_handlers.util import apply_voi_lut


def load_test_transforms():
  transforms = T.Compose([
    T.ToTensor(),
    T.Resize(settings.W, antialias=None),
    T.Normalize(
      mean=settings.IMAGENET_MEAN,
      std=settings.IMAGENET_STD
    )
  ])
  return transforms


def timeit(func, repetitions=1):
  def wrapper(*args, **kwargs):
    all_rep = []
    for _ in tqdm(range(repetitions)):
      tick = time()
      res = func(*args, **kwargs)
      all_rep.append(time() - tick)
    mean_time = sum(all_rep) / len(all_rep)
    print(f"function call to {func.__name__} on average took {round(1000 * mean_time, 5)}ms. (For {repetitions} repetitions)")
    return res
  return wrapper


def fileread(file_like):
  dcm = False
  try:
    dcm_file = pydicom.dcmread(file_like)
    dcm = True
  except InvalidDicomError:
    image = Image.open(file_like)
    image = image.convert("RGB")
    image = np.array(image)

  if dcm:
    image = apply_voi_lut(dcm_file.pixel_array, dcm_file)
    if dcm_file.PhotometricInterpretation == "MONOCHROME1":
      image = np.amax(image) - image
    image = image - np.min(image)
    image = image / np.max(image)
    image = (image * 255).astype(np.uint8)
    image = np.stack([image] * 3, axis=-1)
  return image


def imread(path):
  file_type = path.split(".")[-1]
  assert file_type in settings.ACCEPTED_FILETYPES, f"Invalid file type! Please use one of {settings.ACCEPTED_FILETYPES}"
  if file_type in settings.ACCEPTED_DICOM_FORMATS:
    image = read_dicom(path)
  else:
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  return image 


def read_dicom(path):
  dcm_file = pydicom.dcmread(path)
  # Apply default dicom calibration using the parameters inside the file
  image = apply_voi_lut(dcm_file.pixel_array, dcm_file)

  # Invert f monochrome1
  if dcm_file.PhotometricInterpretation == "MONOCHROME1":
    image = np.amax(image) - image

  # Normalization
  image = image - np.min(image)
  image = image / np.max(image)
  image = (image * 255).astype(np.uint8)
  # Dicom files are usually monochromatic, resulting in 
  # W x H images. Since the model accepts RGB images we 
  # repeat the image on a new axis to create [W x H x 3]
  image = np.stack([image] * 3, axis=-1)
  return image


if __name__ == "__main__":
  import cv2
  path = "../data/sample.dcm"
  image = imread(path)
  print(f"Image Shape: {image.shape}\nImage Max Pixel: {np.max(image)}\nImage Min Pixel: {np.min(image)}")
  cv2.imshow(path, image)
  cv2.waitKey(0)
  cv2.destroyAllWindows()  
