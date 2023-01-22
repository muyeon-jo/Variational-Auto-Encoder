import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import datetime
import os

# show grid image
def imshow_grid(img):
    img = torchvision.utils.make_grid(img)
    print(type(img))
    print(img.shape)
    plt.imshow(img.permute(1, 2, 0))
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    plt.show()   

# VAE model
class VAE(nn.Module):
    def __init__(self, image_size, hidden_size_1, latent_size):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(image_size, hidden_size_1)
        self.fc21 = nn.Linear(hidden_size_1, latent_size)
        self.fc22 = nn.Linear(hidden_size_1, latent_size)

        self.fc3 = nn.Linear(latent_size, hidden_size_1)
        self.fc4 = nn.Linear(hidden_size_1, image_size)

    def encode(self, x):
        h1 = F.tanh(self.fc1(x))
        
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z):
        h3 = F.tanh(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction = 'sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE, KLD
def train(epoch, model, train_loader, optimizer):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(DEVICE)
        optimizer.zero_grad()

        recon_batch, mu, logvar = model(data)

        BCE, KLD = loss_function(recon_batch, data, mu, logvar)

        loss = BCE + KLD

        writer.add_scalar("Train/Reconstruction Error", BCE.item(), batch_idx + epoch * (len(train_loader.dataset)/BATCH_SIZE) )
        writer.add_scalar("Train/KL-Divergence", KLD.item(), batch_idx + epoch * (len(train_loader.dataset)/BATCH_SIZE) )
        writer.add_scalar("Train/Total Loss" , loss.item(), batch_idx + epoch * (len(train_loader.dataset)/BATCH_SIZE) )

        loss.backward()

        train_loss += loss.item()

        optimizer.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))
            
    print("======> Epoch: {} Average loss: {:.4f}".format(
        epoch, train_loss / len(train_loader.dataset)
    ))  

def test(epoch, model, test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(test_loader):
            data = data.to(DEVICE)
            
            recon_batch, mu, logvar = model(data)
            BCE, KLD = loss_function(recon_batch, data, mu, logvar)

            loss = BCE + KLD

            writer.add_scalar("Test/Reconstruction Error", BCE.item(), batch_idx + epoch * (len(test_loader.dataset)/BATCH_SIZE) )
            writer.add_scalar("Test/KL-Divergence", KLD.item(), batch_idx + epoch * (len(test_loader.dataset)/BATCH_SIZE) )
            writer.add_scalar("Test/Total Loss" , loss.item(), batch_idx + epoch * (len(test_loader.dataset)/BATCH_SIZE) )
            test_loss += loss.item()

            if batch_idx == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n], recon_batch.view(BATCH_SIZE, 1, 28, 28)[:n]]) # (16, 1, 28, 28)
                grid = torchvision.utils.make_grid(comparison.cpu()) # (3, 62, 242)
                writer.add_image("Test image - Above: Real data, below: reconstruction data", grid, epoch)
def test_other_input(model,test_data):
    with torch.no_grad():
        test_data=torch.from_numpy(test_data).float()
        test_data=test_data.to(DEVICE)
        mu, logvar = model.encode(test_data)
        print("mean:")
        print(mu)
        print("logvar:")
        print(logvar)
        recon_image = model.decode(model.reparameterize(mu,logvar)).cpu()

        print("recon_image:")
        print(recon_image)
        grid = torchvision.utils.make_grid(recon_image.view(1, 1, 28, 28))
        writer.add_image("test other", grid, 1)  

def latent_to_image(epoch, model):
    with torch.no_grad():
        sample = torch.randn(64, 2).to(DEVICE)
        recon_image = model.decode(sample).cpu()
        grid = torchvision.utils.make_grid(recon_image.view(64, 1, 28, 28))
        writer.add_image("Latent To Image", grid, epoch)                        
if __name__ == "__main__":
    USE_CUDA = torch.cuda.is_available()
    DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
    print("사용하는 Device : ", DEVICE)

    current_time = datetime.datetime.now() + datetime.timedelta()
    current_time = current_time.strftime('%Y-%m-%d-%H_%M')

    saved_loc = os.path.join('./content/VAE_Result/'+current_time)
    os.mkdir(saved_loc)

    print("저장 위치: ", saved_loc)

    writer = SummaryWriter(saved_loc)
    EPOCHS = 50
    BATCH_SIZE = 200
    # Transformer code
    transformer = transforms.Compose([transforms.ToTensor()])

    # Loading trainset, testset and trainloader, testloader
    trainset = torchvision.datasets.MNIST(root = './content/MNIST', train = True,
                                            download = True, transform = transformer)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 2)


    testset = torchvision.datasets.MNIST(root = './content/MNIST', train = False,
                                            download = True, transform = transformer)

    testloader = torch.utils.data.DataLoader(testset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 2)
    # sample check
    sample, label = next(iter(trainloader))

    #imshow_grid(sample[0:8])

    VAE_model = VAE(28*28, 512, 2).to(DEVICE)
    optimizer = optim.Adam(VAE_model.parameters(), lr = 1e-3)
    #test_other_input(VAE_model, np.ones(28*28))
    for epoch in tqdm(range(0, EPOCHS)):
        train(epoch, VAE_model, trainloader, optimizer)
        test(epoch, VAE_model, testloader)
        print("\n")
        latent_to_image(epoch, VAE_model)
    test_other_input(VAE_model, np.ones(28*28))
    
    writer.close()
    