import numpy as np
import time

from CNN_Baseline import CNN
from torch import optim
import torch.nn as nn
import torch
from utils import get_handwritten_batch, get_handwritten_keys, evaluateLabels, mnist_to_files

#add mnist images to dataset folder
mnist_to_files()


# Define the model
model = CNN()
model.train()
# Define loss function and optimizer
# Implement this
loss_fun = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)



batch_count = 5000
for batch_id in range(300): #int(batch_count*0.8)
    if batch_id % 100 == 0:
        print('Batch_id: {}'.format(batch_id))
    labels, images = get_handwritten_batch(batch_count, batch_id)
    images = np.reshape(images, [images.shape[0], 1, images.shape[1], images.shape[2]])
    images = torch.tensor(images).float()
    labelsNum = get_handwritten_keys(labels)
    labelsNum = torch.tensor(labelsNum).long()
    for batch in range(10):
        rand_sample_inds = np.random.randint(low=0, high=images.shape[0]-1, size=images.shape[0]//10)
        rand_sample_x = images[rand_sample_inds]
        rand_sample_y = labelsNum[rand_sample_inds]

        outputs = model(rand_sample_x)
        optimizer.zero_grad()
        # print(outputs.shape)
        # print(rand_sample_y.shape)
        losses = loss_fun(outputs, rand_sample_y)
        losses.backward()
        optimizer.step()

# Test model
yTest, xTest = get_handwritten_batch(batch_count, int(batch_count*0.8), int(batch_count*0.9))
xTest = np.reshape(xTest, [xTest.shape[0], 1, xTest.shape[1], xTest.shape[2]])[:10000]
xTest = torch.tensor(xTest).float()
yTest_num = get_handwritten_keys(yTest)
yTest_num = torch.tensor(yTest_num).long()[:10000]
#  y_validate, x_validate =  get_handwritten_batch(batch_count, int(batch_count*0.9), batch_count)


yPred = np.zeros(yTest.shape[0])
model.eval()  # Set this to evaluation mode
# Loop over xTest and compute labels (implement this)
yPred = torch.argmax(model(xTest), dim=1).numpy()

# Map it back to numpy to use our functions
yTest_num = yTest_num.numpy()
(acc, conf) = evaluateLabels(yTest_num, yPred, False)
print('Accuracy {:.2f} %\n'.format(acc*100))


#Save model
# torch.save(model.state_dict(), "CNN_Baseline.pt")