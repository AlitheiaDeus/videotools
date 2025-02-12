import cv2
import numpy as np
import math
import torch

github_repo = "intel-isl/MiDaS"
print(torch.hub.list(github_repo))

midas_transforms = torch.hub.load(github_repo, "transforms")
print(dir(midas_transforms))
transform = midas_transforms.dpt_transform

device = torch.device("cuda")

model_type = "DPT_BEiT_L_384"
model = torch.hub.load(github_repo, model_type, pretrained=False)
model.to(device)