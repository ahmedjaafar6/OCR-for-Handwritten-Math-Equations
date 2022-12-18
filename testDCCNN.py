import numpy as np
import time

from DCCNN import DCCNN
from torch import optim
import torch.nn as nn
import torch
from utils import get_handwritten_batch, get_handwritten_keys, evaluateLabels, mnist_to_files


#add mnist images to dataset folder
mnist_to_files()


# Define the model
model = DCCNN()
model.train()
# Define loss function and optimizer
# Implement this
loss_fun = nn.CrossEntropyLoss()
optimizer = optim.Adadelta(model.parameters(), lr=0.9, rho=0.95, eps=1e-10)
# optimizer = optim.Adam(model.parameters(), lr=0.01)


start = time.time()
batch_count = 4500
mini_batch_iter = 10
for batch_id in range(int(batch_count*0.8)):
    if batch_id % 50 == 0:
        print('Batch_id: {}'.format(batch_id), end='\r')
    labels, images = get_handwritten_batch(batch_count, batch_id)
    images = np.reshape(images, [images.shape[0], 1, images.shape[1], images.shape[2]])
    images = torch.tensor(images).float()
    labelsNum = get_handwritten_keys(labels)
    labelsNum = torch.tensor(labelsNum).long()
    for batch in range(mini_batch_iter):
        rand_sample_inds = np.random.randint(low=0, high=images.shape[0]-1, size=images.shape[0]//mini_batch_iter)
        rand_sample_x = images[rand_sample_inds]
        rand_sample_y = labelsNum[rand_sample_inds]

        outputs = model(rand_sample_x)
        optimizer.zero_grad()
        losses = loss_fun(outputs, rand_sample_y)
        losses.backward()
        optimizer.step()


# Validate Model
yVal, xVal = get_handwritten_batch(batch_count, int(batch_count*0.8), int(batch_count*0.9))
xVal = np.reshape(xVal, [xVal.shape[0], 1, xVal.shape[1], xVal.shape[2]])
xVal = torch.tensor(xVal).float()
yVal_num = get_handwritten_keys(yVal)
yVal_num = torch.tensor(yVal_num).long()

yPred = np.array([])
model.eval()  # Set this to evaluation mode
with torch.no_grad():
    batches = np.split(xVal, xVal.shape[0]//800)
    for b in batches:
        yPred_b = torch.argmax(model(b), dim=1).numpy()
        yPred = np.concatenate((yPred, yPred_b))

# Map it back to numpy
yVal_num = yVal_num.numpy()
(acc, conf) = evaluateLabels(yVal_num, yPred, False)
print('Validation Accuracy {:.2f} %\n'.format(acc*100))




#Test model
yTest, xTest = get_handwritten_batch(batch_count, int(batch_count*0.9), int(batch_count))
xTest = np.reshape(xTest, [xTest.shape[0], 1, xTest.shape[1], xTest.shape[2]])
xTest = torch.tensor(xTest).float()
yTest_num = get_handwritten_keys(yTest)
yTest_num = torch.tensor(yTest_num).long()


yPred = np.array([])
model.eval()  # Set this to evaluation mode
with torch.no_grad():
    batches = np.split(xTest, xTest.shape[0]//800)
    for b in batches:
        yPred_b = torch.argmax(model(b), dim=1).numpy()
        yPred = np.concatenate((yPred, yPred_b))


# # Map it back to numpy
yTest_num = yTest_num.numpy()
(acc, conf) = evaluateLabels(yTest_num, yPred, False)
print('Testing Accuracy {:.2f} %\n'.format(acc*100))

end = time.time()
print((end - start)/60)

#Save model
torch.save(model.state_dict(), "DCCNN.pt")