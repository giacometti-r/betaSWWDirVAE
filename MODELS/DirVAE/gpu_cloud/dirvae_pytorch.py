import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.cuda.amp import autocast, GradScaler
torch.manual_seed(42)

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()

        self.dense1 = nn.Linear(784, 500)
        self.dense2 = nn.Linear(500, 500)
        self.dense3 = nn.Linear(500, latent_dim)

    def sample(self, alpha_hat):
        u = torch.rand(size=alpha_hat.size(), requires_grad=True).to(device)
        v = torch.pow(u * alpha_hat * torch.exp(torch.lgamma(alpha_hat)),1.0/alpha_hat)
        z = v / torch.sum(v)
        return z

    def forward(self, x):
        alpha_hat = x.view(-1, 28*28)
        alpha_hat = F.relu(self.dense1(alpha_hat))
        alpha_hat = F.relu(self.dense2(alpha_hat))
        alpha_hat = F.softplus(self.dense3(alpha_hat))
        z = self.sample(alpha_hat)
        return z, alpha_hat

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.dense1 = nn.Linear(latent_dim, 500)
        self.dense2 = nn.Linear(500, 28*28)

    def forward(self, x):
        x_hat = F.relu(self.dense1(x))
        return self.dense2(x_hat)
    

class DirVAE(nn.Module):
    def __init__(self, latent_dim):
        super(DirVAE, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
    
    def forward(self, x):
        z, alpha_hat = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, alpha_hat
    
def ELBO(x_hat, x, alpha_hat, alpha):
    
    likelihood = F.binary_cross_entropy_with_logits(x_hat, x.view(-1, 28*28), reduction='sum')
    
    lgamma_alpha = torch.lgamma(alpha).to(device)
    lgamma_alpha_hat = torch.lgamma(alpha_hat).to(device)
    digamma_alpha_hat = torch.digamma(alpha_hat).to(device)
    
    kld = torch.sum(lgamma_alpha - lgamma_alpha_hat + (alpha_hat - alpha) * digamma_alpha_hat)
    
    # if torch.isnan(likelihood):
    #     print('LIKELIHOOD IS NAN')
        
    # if torch.isnan(kld):
    #     print('KLD IS NAN') 

    return likelihood + kld


train_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=True, download=True, transform=transforms.ToTensor()),
                                           batch_size=100, shuffle=True)
test_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
                                          batch_size=100, shuffle=True)
cuda = torch.cuda.is_available()

print('CUDA:', cuda)

device = torch.device("cuda" if cuda else "cpu")

latent_dim = 50

model = DirVAE(latent_dim).to(device)

params = model.parameters()
optimizer = optim.Adam(params, lr=5e-4)

alpha =  ((1 - 1/latent_dim) * torch.ones(size=(latent_dim,))).to(device)

epochs = 1000

scaler = GradScaler()

train_total_loss = []
val_total_loss = []

for epoch in range(epochs):
    model.train()
    for batch_idx, (x, _) in enumerate(train_loader): 
        x = x.to(device)
        optimizer.zero_grad()
        with autocast():
            x_hat, alpha_hat = model(x)
            loss = ELBO(x_hat, x, alpha_hat, alpha)
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(optimizer)
        scaler.update()
    print(f'loss at end of epoch {epoch}: {loss.item()}')
    
    train_total_loss.append(loss.to('cpu').numpy())

    model.eval()
    with torch.no_grad():
        for i, (val_x, _) in enumerate(test_loader):
            val_x = val_x.to(device)
            val_x_hat, val_alpha_hat = model(val_x)
            test_loss = ELBO(val_x_hat, val_x, val_alpha_hat, alpha)
    print(f'test loss at end of epoch {epoch}: {test_loss.item()}')

    val_total_loss.append(test_loss.to('cpu').numpy())
    
    if epoch == 0:
        plt.imshow(test_loader.dataset[0][0].numpy().reshape(28,28))
        plt.savefig('/reconstructed_images/original.png')
    with torch.no_grad():
        sample = test_loader.dataset[0][0].to(device)
        img, img_alpha_hat = model(sample)
    img = torch.sigmoid(img)
    img = img.to('cpu').numpy().reshape(28,28)
    plt.imshow(img)
    plt.title(f'epoch_{epoch}')
    plt.savefig(f'/reconstructed_images/epoch_{epoch}.png')

    torch.save(model.state_dict(), '/weights/model.pth')

df_loss = pd.DataFrame(np.vstack([train_total_loss, val_total_loss]).T, columns=['train_loss', 'val_loss'])
df_loss.to_csv('loss.csv', index=False)
