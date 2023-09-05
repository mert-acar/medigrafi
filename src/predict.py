from model import Medigrafi 


def preload_model():
  model = Medigrafi(
    pretrained="../checkpoints/densenet_weights",
    postprocessed=True
  )
  model.eval()
  return model


if __name__ == "__main__":
  import cv2
  from utils import timeit, imread

  # Load the image
  image = imread("../data/cardiomegaly.png")

  # Load the model
  model = preload_model()

  # Test model performance
  # predict = timeit(model.predict, 1000)
  # probs, heatmap = predict(image, heatmap=True)
  # probs = predict(image)

  # Run inference on the model
  probs, heatmap = model.predict(image, heatmap=True)

  # Display the results
  cv2.imshow("Heatmap", heatmap)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
