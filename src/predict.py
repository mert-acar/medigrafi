import cv2
from model import Medigrafi
import matplotlib.pyplot as plt
from utils import imread, load_test_transforms

# Load the image
image = imread("../data/cardiomegaly.png")

# Load the model
model = Medigrafi("../checkpoints/densenet_weights")
model.eval() # very important

transform = load_test_transforms()

tensor_image = transform(image).unsqueeze(0)
# Run inference on the model
probs = model(tensor_image)
heatmap = model.create_heatmap("Cardiomegaly")

heatmap = cv2.resize(heatmap, image.shape[:-1])
layered = ((image / 255) * 0.5) + (heatmap * 0.5)
print(probs)

# Display the results
plt.imshow(layered)
plt.show()
