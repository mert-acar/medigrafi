import torch
from model import Medigrafi 
from torchvision import transforms as T


def predict(image):
  """
    Supply the RGB image as an argument to run prediction on it. Examples:
    '''
      from PIL import Image
      image = Image.open("/path/to/image")
      image = image.convert("RGB")
    '''
    
    or

    '''
      import cv2
      image = cv2.imread("/path/to/image")
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    '''
  """
  trans = T.Compose([
    T.Resize(224), # Resize to 224 x 244 x 3
    T.ToTensor(),  # Create a tensor 3 x 224 x 224
    T.Normalize(   # Normalize the RGB values with imagenet statistics to get standard images
      mean=(0.485, 0.456, 0.406),
      std=(0.229, 0.224, 0.225)
    )
  ])
  
  image = trans(image).unsqueeze(0) # [3, 224, 224] -> [1, 3, 224, 224] (1 being the batch size, can be N)
  model = Medigrafi(
    pretrained="../checkpoints/densenet_weights",
    postprocessed=True
  )
  model.eval()

  _, o = model(image) # Run the prediction
  return o
