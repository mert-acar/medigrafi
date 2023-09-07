import cv2
import settings
import streamlit as st
from model import Medigrafi
from utils import fileread, load_test_transforms

def reset():
  st.session_state.image = None
  st.session_state.probs = None


if __name__ == "__main__":
  st.set_page_config(
    layout = "wide",
    page_title = "Medigrafi - AI X-Ray Diagnostic Companion"
  )
  
  # Need session state for stateful execution
  ss = st.session_state

  # Preload the model
  if "model" not in ss:
    ss.model = Medigrafi("../checkpoints/densenet_weights")
    ss.model.eval()
  if "transform" not in ss:
    ss.transform = load_test_transforms()
  if "image" not in ss:
    ss.image = None
  if "probs" not in ss:
    ss.probs = None
  
  _, center, _ = st.columns((0.15, 0.7, 0.15))
  center.image("../data/banner.png")

  upload = center.file_uploader("Upload your file here", type=settings.ACCEPTED_FILETYPES, on_change=reset)
  if upload is not None:
    ss.image = fileread(upload)
    with center.status("Processing your file...", expanded=True) as status:
      tensor_image = ss.transform(ss.image).unsqueeze(0)
      if ss.probs is None:
        ss.probs = ss.model(tensor_image)
        ss.probs = {key: round(value.item(), 4) for (key, value) in zip(settings.LABELS, ss.probs[0])}
        ss.probs = dict(sorted(ss.probs.items(), key=lambda x: x[1], reverse=True))
        ss.probs = dict(filter(lambda x: x[1] > settings.PREDICTION_THRESHOLD, ss.probs.items()))
      status.update(label="Inference done! :white_check_mark:", expanded=True, state='complete')
      if len(ss.probs) > 0:
        c1, c2 = st.columns((0.2, 0.8))
        c1.markdown("#")
        c1.markdown("#")
        c1.markdown("#")
        label = c1.radio(
          "# Diagnosis",
          list(ss.probs.keys()),
          captions=["***Score***: " + str(round(val * 100, 3)) for val in ss.probs.values()]
        )
        opacity = c2.slider("Heatmap Opacity (%)", min_value=0, max_value=100, value=50, step=10)
        opacity = opacity / 100
        heatmap = cv2.resize(
          ss.model.create_heatmap(label),
          ss.image.shape[:-1]
        )
        layered = ((1 - opacity) * (ss.image / 255)) + (opacity * heatmap)
        c2.image(layered)
      else:
        st.success("No disease found!")
        st.image(ss.image)
  else:
    ss.probs = None

