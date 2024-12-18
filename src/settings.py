ACCEPTED_IMAGE_FORMATS = ["jpg", "png", "jpeg"]
ACCEPTED_DICOM_FORMATS = ["dcm", "dicom"]
ACCEPTED_FILETYPES = ACCEPTED_IMAGE_FORMATS + ACCEPTED_DICOM_FORMATS

W = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

PREDICTION_THRESHOLDS = {
  "Atelectasis": 0.45,
  "Cardiomegaly": 0.5,
  "Effusion": 0.6,
  "Infiltration": 0.4,
  "Mass": 0.5,
  "Nodule": 0.5,
  "Pneumonia": 0.45,
  "Pneumothorax": 0.3,
  "Consolidation": 0.5,
  "Edema": 0.25,
  "Emphysema": 0.8,
  "Fibrosis": 0.55,
  "Pleural_Thickening": 0.85,
  "Hernia": 0.8,
}

PP_WEIGHTS = [
  0.61, 0.61, 0.58, 0.58, 0.31, 0.5, 0.68,
  0.62, 0.9, 0.63, 0.41, 0.63, 0.77, 0.45
]

PP_BASELINE = [
  0.103, 0.025, 0.119, 0.177, 0.051, 0.056, 0.012,
  0.047, 0.042, 0.021, 0.022, 0.015, 0.030, 0.002
]


LABELS = [
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
