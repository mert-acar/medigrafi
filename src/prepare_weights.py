import re
import torch
from tqdm import tqdm
from collections import OrderedDict

# state = torch.load("../checkpoints/checkpoint_densenet")['model'].state_dict()
state = torch.load("../checkpoints/densenet_state_dict")
weights = {'features': OrderedDict(), 'classifier': OrderedDict()}
for key, value in tqdm(state.items()):
  if "classifier" not in key:
    key = key.replace("features.", "")
    updated_key = re.sub(r'\.(\d+)', r'\1', key)
    weights['features'][updated_key] = value
  else:
    updated_key = re.sub(r'.*\.', '', key)
    weights['classifier'][updated_key] = value
torch.save(weights, "../checkpoints/densenet_weights")
