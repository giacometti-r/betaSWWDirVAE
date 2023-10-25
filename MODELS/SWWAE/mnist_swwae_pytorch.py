import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import tqdm

# Define the residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=(kernel_size-1)//2)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        return out

# Define the encoder and decoder
class Autoencoder(nn.Module):
    def __init__(self, nfeats):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            ResidualBlock(1, nfeats[0]),
            nn.MaxPool2d(2),
            ResidualBlock(nfeats[0], nfeats[1]),
            nn.MaxPool2d(2),
            ResidualBlock(nfeats[1], nfeats[2]),
            nn.MaxPool2d(2),
            ResidualBlock(nfeats[2], nfeats[3]),
            nn.MaxPool2d(2),
            ResidualBlock(nfeats[3], nfeats[4])
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(nfeats[4], nfeats[3], kernel_size=3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(nfeats[3], nfeats[2], kernel_size=3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(nfeats[2], nfeats[1], kernel_size=3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(nfeats[1], nfeats[0], kernel_size=3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(nfeats[0], 1, kernel_size=3, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=128, shuffle=False)

# Initialize autoencoder
autoencoder = Autoencoder(nfeats=[8, 16, 32, 64, 128])
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

# Train autoencoder
for epoch in range(10):
    for batch in tqdm.tqdm(train_loader):
        images, _ = batch
        optimizer.zero_grad()
        outputs = autoencoder(images)
        loss = criterion(outputs, images)
        loss.backward()
        optimizer.step()

# Test autoencoder
with torch.no_grad():
    for batch in test_loader:
        images, _ = batch
        reconstructions = autoencoder(images)
        break

# Plot results
import matplotlib.pyplot as plt

def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')

imshow(torchvision.utils.make_grid(images[:25]))
plt.show()

imshow(torchvision.utils.make_grid(reconstructions[:25]))
plt.show()

print('Done')
