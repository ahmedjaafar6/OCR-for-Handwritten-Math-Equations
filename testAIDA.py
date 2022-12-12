from CNN_Baseline import CNN
import torch
from utils import get_aida_batch, standardize_image
import numpy as np
from main import Parser



images = []
batch = get_aida_batch(1)
for i in range(10):
    images.append(next(batch))
images = np.array(images)

# Test with AIDA
model = CNN()
model.load_state_dict(torch.load("CNN_Baseline.pt"))
model.eval()

p = Parser(images[0], 0.005)