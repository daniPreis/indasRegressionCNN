import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import ImageAnglesDataset
from regressionNet import net

img_dir = 'D:/Dokumente/Uni/Master/Indas/BMW - Material-20210512/images'
new_csv_dir = 'D:/Dokumente/Uni/Master/Indas/BMW - Material-20210512/Codex_train_new.csv'

criterion = nn.MSELoss(reduction='mean')
optimizer = optim.Adam(net.parameters(), lr=1E-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

transform = transforms.Compose([
    transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

dataset = ImageAnglesDataset(csv_file=new_csv_dir,
                             root_dir=img_dir)
angles_dataset, angles_dataset_test = torch.utils.data.random_split(dataset, [3000, 2250])

data_loader = DataLoader(angles_dataset, batch_size=10, shuffle=True)

# net.load_state_dict(torch.load('weights.pt'))

for epoch in range(3):

    # set the running loss at each epoch to zero
    running_loss = 0.0
    # we will enumerate the train loader with starting index of 0
    # for each iteration (i) and the data (tuple of input and labels)
    for i, data in enumerate(data_loader):

        # inputs, labels = data
        inputs = data['image'].float()
        labels = data['Angle'].float()
        # clear the gradient
        optimizer.zero_grad()
        # feed the input and acquire the output from network

        outputs = net(inputs)
        # calculating the predicted and the expected loss
        loss = criterion(outputs, labels)

        # compute the gradient
        loss.backward()

        # update the parameters
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 10 == 0:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0

torch.save(net.state_dict(), 'D:/Dokumente/Uni/Master/Indas/BMW - Material-20210512/weights.pt')


data_loader_test = DataLoader(angles_dataset_test, batch_size=len(angles_dataset_test))

test_iterator = iter(data_loader_test)
data = test_iterator.next()
images = data['image'].float()
labels = data['Angle'].detach().numpy()

results = net(images).detach().numpy()
np.set_printoptions(suppress=True)
test_test = np.concatenate((labels, results), 1) * 360
print(test_test)
error = np.sum(np.subtract(labels, results) ** 2, 0) / len(labels)
pd.DataFrame(test_test, columns=['labels', 'results']).to_csv(
    'D:/Dokumente/Uni/Master/Indas/BMW - Material-20210512/test_res.csv')
print(error)
