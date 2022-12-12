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

p = Parser(images[0], 0.005, show=False)
i = np.array(p[0]).astype(np.float32)
i = standardize_image(i, square=True)

i = np.reshape(i, [1, 1,i.shape[0], i.shape[1]])
i = torch.tensor(i).float()
yPred = torch.argmax(model(i), dim=1).numpy()

print(yPred)
