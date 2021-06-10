import torch.nn as nn

# sequential based model
'''
seq_model = nn.Sequential(
    nn.Conv2d(3, 10, kernel_size=5),
    nn.Conv2d(10, 10, kernel_size=5),
    nn.Conv2d(10, 10, kernel_size=5),
    nn.Conv2d(10, 10, kernel_size=5),
    nn.Conv2d(10, 1, kernel_size=5),
    nn.Flatten(),
    nn.Linear(144400, 320),
    nn.ReLU(),
    nn.Linear(320, 50),
    nn.ReLU(),
    nn.Linear(50, 1),
    nn.ReLU()
)
'''

seq_model = nn.Sequential(
    nn.Conv2d(3, 96, stride=4, kernel_size=11),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.BatchNorm2d(96),
    nn.Conv2d(96, 256, stride=1, kernel_size=5, padding=2),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.BatchNorm2d(256),

    nn.Conv2d(256, 384, stride=1, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Conv2d(384, 384, stride=1, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Conv2d(384, 256, stride=1, kernel_size=3, padding=1),
    nn.ReLU(),

    nn.MaxPool2d(kernel_size=3, stride=2),

    nn.Dropout2d(0.5),
    nn.Flatten(), # Next three rows fully connected layer
    nn.Linear(30976, 4096),
    nn.ReLU(),

    nn.Dropout(0.5),
    nn.Linear(4096, 4096),
    nn.ReLU(),

    nn.Linear(4096, 1),
    nn.ReLU(),
)

net = seq_model
