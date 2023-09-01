import cv2
import pydicom
import numpy as np
from pydicom.pixel_data_handlers.util import apply_voi_lut

def read_dicom(path):
  dcm_file = pydicom.dcmread(path)
  image = apply_voi_lut(dcm_file.pixel_array, dcm_file)
  if dcm_file.PhotometricInterpretation == "MONOCHROME1":
    image = np.amax(image) - image
  image = image - np.min(image)
  image = image / np.max(image)
  image = (image * 255).astype(np.uint8)
  return image


if __name__ == "__main__":
  path = "../data/sample.dcm"
  image = read_dicom(path)
  print(f"Image Shape: {image.shape}\nImage Max Pixel: {np.max(image)}\nImage Min Pixel: {np.min(image)}")
  cv2.imshow(path, image)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
