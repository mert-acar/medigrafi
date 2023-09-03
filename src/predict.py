import settings
from pprint import pprint
from model import Medigrafi 
from utils import create_heatmap
from torchvision import transforms as T

def preload():
  trans = T.Compose([
    T.ToTensor(),                   # Create a tensor 3 x W x H
    T.Resize(settings.W, antialias=None),  # Resize to 224 x 224 x 3
    T.Normalize(                    # Normalize the RGB values with imagenet statistics to get standard images
      mean=settings.IMAGENET_MEAN,
      std=settings.IMAGENET_STD
    )
  ])

  model = Medigrafi(
    pretrained="../checkpoints/densenet_weights",
    postprocessed=True
  )
  model.eval()
  return trans, model


def predict(trans, model, image, heatmap=False, disease=None):
  """
    Parameters
    ==========
    + image:
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
      
      or
      
      '''
        from utils import read_dicom 
        image = read_dicom("/path/to/dicom", model_ready=True)
      '''

    + heatmap:
      The heatmap for the selected disease. If no disease is specified the
      heatmap is created for the most probable disease above the positve treshold

    + disase:
      Disease label for which the heatmap is going to be created. Pass 'None' for 
      the most probable positive disease


    Returns
    =======
    The output probabilities and optionally the heatmap for given disease

  """
  
  image = trans(image).unsqueeze(0) # [3, 224, 224] -> [1, 3, 224, 224] (1 being the batch size, can be N)
  features, probs = model(image)
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
  if heatmap:
    if disease is not None and disease in [x[0] for x in pos_preds]:
      label = disease
    elif len(pos_preds) > 0:
      label = pos_preds[0][0]
    else:
      pprint(preds)
      raise RuntimeError("The diagnosis for given label is not positive!")
    weight, bias = model.export_weights_for_heatmap()
    heatmap = create_heatmap(image[0], label, features[0], weight, bias)
    return pos_preds, heatmap
  return pos_preds
  


if __name__ == "__main__":
  import cv2
  # from utils import read_dicom
  # image = read_dicom("../data/sample.dcm", model_ready=True)
  image = cv2.imread("../data/cardiomegaly.png")
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  trans, model = preload()
  probs, heatmap = predict(trans, model, image, True)
  cv2.imshow("Heatmap", heatmap)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
