import numpy as np
import os
import time

from CNNModel import CNN
from torch import optim
import torch.nn as nn
import torch

# Accuracy placeholder
accuracy = np.zeros(len(dataTypes))

#Load data
path = os.path.join('..', 'data', dataType)
data = utils.loadmat(path)
print('+++ Loading dataset: {} ({} images)'.format(dataType, data['x'].shape[2]))

# Organize into numImages x numChannels x width x height
x = data['x'].transpose([2,0,1])
x = np.reshape(x,[x.shape[0], 1, x.shape[1], x.shape[2]])
y = data['y']
# Convert data into torch tensors
x = torch.tensor(x).float()
y = torch.tensor(y).long() # Labels are categorical

# Define the model
model = CNN()
model.train()
# Define loss function and optimizer
# Implement this
loss_fun = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)

# Start training
xTrain = x[data['set']==trainSet,:,:,:]
yTrain = y[data['set']==trainSet]


# Loop over training data in some batches
# Implement this
epochs = 1000
batch_size = 64
for j in range(epochs):
    rand_sample_inds = np.random.randint(low=0, high=xTrain.shape[0]-1, size=batch_size)
    rand_sample_x = xTrain[rand_sample_inds]
    rand_sample_y = yTrain[rand_sample_inds]
    outputs = model(rand_sample_x)
    optimizer.zero_grad()
    losses = loss_fun(outputs, rand_sample_y)
    losses.backward()
    optimizer.step()
    
# Test model
xTest = x[data['set']==testSet,:,:,:]
yTest = y[data['set']==testSet]

yPred = np.zeros(yTest.shape[0])
model.eval() # Set this to evaluation mode
# Loop over xTest and compute labels (implement this)
yPred = torch.argmax(model(xTest), dim=1).numpy()

# Map it back to numpy to use our functions
yTest = yTest.numpy()
(acc, conf) = utils.evaluateLabels(yTest, yPred, False)
print('Accuracy [testSet={}] {:.2f} %\n'.format(testSet, acc*100))
accuracy[i] = acc

# Print the results in a table
print('+++ Accuracy Table [trainSet={}, testSet={}]'.format(trainSet, testSet))
print('--------------------------------------------------')
print('dataset\t\t\t', end="")
print('{}\t'.format('cnn'), end="")
print()
print('--------------------------------------------------')
for i in range(len(dataTypes)):
    print('{}\t'.format(dataTypes[i]), end="")
    print('{:.2f}\t'.format(accuracy[i]*100))

# Once you have optimized the hyperparameters, you can report test accuracy
# by setting testSet=3. Do not optimize your hyperparameters on the
# test set. That would be cheating.
