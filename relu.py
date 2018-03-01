from random import randint

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

nrows = 9000
ntrain = int(nrows * .7)
X = torch.rand(nrows, 3)
Y = torch.mm(X, torch.from_numpy(
    np.array([[.1], [2], [3]]).astype(np.float32)))
# concat two tensors, like hstack in numpy
# Y = torch.cat([Y < torch.mean(Y), Y >= torch.mean(Y)], dim=1).type(torch.LongTensor)
Y = (Y >= torch.mean(Y)).type(torch.LongTensor).view(nrows)
Xtr = X[:ntrain, :]
Ytr = Y[:ntrain]
Xte = X[ntrain:, :]
Yte = Y[ntrain:]

grad_dict: dict = {}


def fc_hook(layer_name, grad_output, log_every_n_step: int):
    n = randint(0, log_every_n_step)
    if n == 0:
        if layer_name in grad_dict:
            batch_mean = torch.abs(
                torch.mean(grad_output[0], 0)
            ).cpu().numpy()
            grad_dict[layer_name].append(batch_mean)
        else:
            grad_dict[layer_name] = []


class LinearWithID(nn.Linear):
    def __init__(self, id, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias=bias)
        self.id = id

    def __repr__(self):
        return self.id


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hooked = False
        self.fc1 = LinearWithID("fc1", 3, 10)
        self.fc2 = LinearWithID("fc2", 10, 10)
        self.fc3 = LinearWithID("fc3", 10, 10)
        self.fc4 = LinearWithID("fc4", 10, 10)
        self.fc5 = LinearWithID("fc5", 10, 10)
        self.fc6 = LinearWithID("fc6", 10, 10)
        self.fc7 = LinearWithID("fc7", 10, 2)
        self.fc1_hook_handle = self.fc1.register_backward_hook(self.backward_hook)
        self.fc2_hook_handle = self.fc2.register_backward_hook(self.backward_hook)
        self.fc3_hook_handle = self.fc3.register_backward_hook(self.backward_hook)
        self.fc4_hook_handle = self.fc4.register_backward_hook(self.backward_hook)
        self.fc5_hook_handle = self.fc5.register_backward_hook(self.backward_hook)
        self.fc6_hook_handle = self.fc6.register_backward_hook(self.backward_hook)
        self.fc7_hook_handle = self.fc7.register_backward_hook(self.backward_hook)
        self.log_every_n_step = 50

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)
        x = F.relu(x)
        x = self.fc6(x)
        x = F.relu(x)
        x = self.fc7(x)
        return x

    def backward_hook(self, module, grad_input,
                      grad_output):  # grad_input is input data of last op of the layer, ignored here
        fc_hook(module.id, grad_output, self.log_every_n_step)


net = Net().cuda()
print(net)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=.8)
NUM_EPOCH = 2
NUM_PER_BATCH = 4

# # one pass backprop
# index_pool = np.arange(Xtr.size(0))
# indices = np.random.choice(index_pool, size=NUM_PER_BATCH, replace=False)
# inputs = Xtr[indices, :].cuda()
# labels = Ytr[torch.from_numpy(indices)].cuda()
# inputs, labels = Variable(inputs), Variable(labels)
# outputs = net(inputs)
# optimizer.zero_grad()
# loss = criterion(outputs, labels)
# loss.backward()
# optimizer.step()
# running_loss += loss.data.item()

NUM_EPOCH = 6
NUM_PER_BATCH = 4
index_pool = np.arange(Xtr.size(0))
for epoch in range(NUM_EPOCH):  # loop over the dataset multiple times
    running_loss = 0.0
    for i in index_pool:
        indices = np.random.choice(
            index_pool, size=NUM_PER_BATCH, replace=False)
        inputs = Xtr[indices, :].cuda()
        labels = Ytr[torch.from_numpy(indices)].cuda()
        inputs, labels = Variable(inputs), Variable(labels)
        outputs = net(inputs)
        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.data.item()
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

accuracy = torch.mean(
    torch.eq(
        torch.max(
            net(Variable(Xte.cuda())),
            dim=1
        )[1].cpu(),
        Yte
    ).type(torch.FloatTensor)
)
print("Accuracy of prediction on test dataset: %f" % accuracy.item())


import os
from scipy.misc import imsave

def png_path(layer_name):
    output_base, _ = os.path.splitext(__file__)
    return output_base + layer_name + ".png"



for k, v in grad_dict.items():
    print(k)
    img = np.vstack(v)
    imsave(
        png_path(k),
        img,
    )

