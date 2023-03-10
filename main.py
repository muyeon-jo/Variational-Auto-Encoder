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
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import datetime
import os
import pickleData

# VAE model
class VAE(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, latent_size):
        super(VAE, self).__init__()
        self.input_size= input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.latent_size = latent_size
        self.fc1 = nn.Linear(input_size, hidden_size_1)
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.fc31 = nn.Linear(hidden_size_2, latent_size)
        self.fc32 = nn.Linear(hidden_size_2, latent_size)

        self.fc4 = nn.Linear(latent_size, hidden_size_2)
        self.fc5 = nn.Linear(hidden_size_2, hidden_size_1)
        self.fc6 = nn.Linear(hidden_size_1, input_size)

    def encode(self, x):
        h1 = torch.tanh(self.fc1(x))
        h2 = torch.relu(self.fc2(h1))
        return self.fc31(h2), self.fc32(h2)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z):
        h3 = torch.tanh(self.fc4(z))
        h4 = torch.relu(self.fc5(h3))
        return torch.sigmoid(self.fc6(h4))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_size))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
def loss_function(recon_x, x, mu, logvar, input_size):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, input_size), reduction = 'sum')
    BCE = torch.mean(BCE)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD = torch.mean(KLD)
    return BCE, KLD
def train(epoch, model, train_loader, optimizer, input_size, lam:float = 1.0):
    #????????? ????????? ??????????????? training mode??? ???????????????.
    #????????? ?????? ????????? ?????? ???????????? ????????? ????????? ?????? ??? ????????? ?????? ???????????? ????????? ??????
    model.train()

    train_loss = 0
    batch_avrg_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(DEVICE)
        optimizer.zero_grad()

        recon_batch, mu, logvar = model(data)

        BCE, KLD = loss_function(recon_batch, data, mu, logvar, input_size)

        loss = BCE + (lam) * KLD

        writer.add_scalar("Train/Reconstruction Error", BCE.item(), batch_idx + epoch * (len(train_loader.dataset)/BATCH_SIZE) )
        writer.add_scalar("Train/KL-Divergence", KLD.item(), batch_idx + epoch * (len(train_loader.dataset)/BATCH_SIZE) )
        writer.add_scalar("Train/Total Loss" , loss.item(), batch_idx + epoch * (len(train_loader.dataset)/BATCH_SIZE) )

        #loss??? ????????? ??? loss?????? ????????? ?????????????????? ????????? ?????? ???????????? ?????? ????????? ????????? ?????????
        #??? ????????? ?????? ????????? ????????? ?????? grad?????? ???????????? ????????????.
        loss.backward()

        train_loss += loss.item()
        batch_avrg_loss+=loss.item()
        
        #??????????????? grad?????? ?????? ??????????????? ????????????.
        optimizer.step()

        if (batch_idx+1) % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t Loss: {:.6f}'.format(
                epoch, batch_idx * BATCH_SIZE + len(data), len(train_loader.dataset),
                100. * (batch_idx) / len(train_loader),
                batch_avrg_loss / (BATCH_SIZE * 9 + len(data))))
            batch_avrg_loss = 0
            
    print("======> Epoch: {} Average loss: {:.4f}".format(
        epoch, train_loss / len(train_loader.dataset)
    ))  

def test(epoch, model, test_loader, input_size, lam:float = 1.0):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(test_loader):
            data = data.to(DEVICE)
            
            recon_batch, mu, logvar = model(data)
            BCE, KLD = loss_function(recon_batch, data, mu, logvar,input_size)
            loss = BCE + lam*KLD

            writer.add_scalar("Test/Reconstruction Error", BCE.item(), batch_idx + epoch * (len(test_loader.dataset)/BATCH_SIZE) )
            writer.add_scalar("Test/KL-Divergence", KLD.item(), batch_idx + epoch * (len(test_loader.dataset)/BATCH_SIZE) )
            writer.add_scalar("Test/Total Loss" , loss.item(), batch_idx + epoch * (len(test_loader.dataset)/BATCH_SIZE) )
            test_loss += loss.item()

            if batch_idx == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n], recon_batch.view(BATCH_SIZE,input_len)[:n]]) # (16, 1, 28, 28)
                grid = torchvision.utils.make_grid(comparison.cpu()) # (3, 62, 242)
                writer.add_image("Test image - Above: Real data, below: reconstruction data", grid, epoch)
def getLatentVector(model, input):
    with torch.no_grad():
        mu, logvar = model.encode(input)
        return mu
class CustomDataset(Dataset):
    def __init__(self, data, label):
        self.x_data = data
        self.y_data = label
    
    #????????? ??? ?????? ?????? 
    def __len__(self):
        return len(self.x_data)
    
    #???????????? ???????????? ???????????? tensor??? ?????? 
    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.y_data[idx])
        return x,y              

def makeUserData():
    f = pickleData.pickle_load("./content/POI(philadelphia)/philadelphia10 200m/userVisitDataPerArea.pkl")
    r = pickleData.pickle_load("./content/POI(philadelphia)/philadelphia10 200m/user_id2Index.pkl")
    label = []
    data = []
    #total = []
    print("user length = "+str(len(r)))
    for i,row in f.items():
        for j, v in row.items():
            if len(v) < 1:
                continue
            li = np.zeros(len(r))
            for key, value in v.items():
                li[key] = value
                if value >1.0:
                    print(value)
            
            data.append(li)
            label.append([i,j])

    
    return data , label, len(r)
def makeCateData():
    f = pickleData.pickle_load("./content/POI(philadelphia)/philadelphia10 200m/visitedCategoryPerArea.pkl")
    r = pickleData.pickle_load("./content/POI(philadelphia)/philadelphia10 200m/cate2Index.pkl")
    userlabels = pickleData.pickle_load("./content/Embeddings/userlabel.pkl")
    label = []
    data = []
    """
    ?????? ????????? ??? ???????????? ???????????? ???????????? ?????? ????????? ?????? ????????? ????????? ???????????? ??????????????? ?????????????????? ?????????
    ???????????? ????????? ????????? ????????? ???????????? ?????? ????????? ????????? ???????????? ??????????????? ?????????????????? ??????????????????.
    """
    #????????? ????????? ????????? ?????????????????? ????????????
    labelsDict = dict()
    for i in userlabels:
        labelsDict[str(i[0])+","+str(i[1])] = 1

    print("category length = "+str(len(r)))
    for i,row in f.items():
        for j, v in row.items():
            checker = False
            try:
                labelsDict[str(i)+","+str(j)] +=1
            except:
                checker = True

            if len(v) < 1 and checker:
                continue
            li = np.zeros(len(r))
            sum = 0
            for key, value in v.items():
                sum+=value
            for key, value in v.items():
                li[key] = value/sum
            
            data.append(li)
            label.append([i,j])
    return data , label, len(r)

def getUserEmbedding(model):
    data,label,length  = makeUserData()
    data = CustomDataset(data,label)
    embedding = []
    for i, label in data:
        temp = getLatentVector(model,i.to("cuda"))
        embedding.append(temp)
    pickleData.pickle_save(embedding,"./content/Embeddings/userEmbed.pkl")

def getCategoryEmbedding(model):
    data,label,length  = makeCateData()
    data = CustomDataset(data,label)
    embedding = []
    for i, label in data: 
        temp = getLatentVector(model,i.to("cuda"))
        embedding.append(temp)
    pickleData.pickle_save(embedding,"./content/Embeddings/categoryEmbed.pkl")
if __name__ == "__main__":
    USE_CUDA = torch.cuda.is_available()
    DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
    print("???????????? Device : ", DEVICE)

    current_time = datetime.datetime.now() + datetime.timedelta()
    current_time = current_time.strftime('%Y-%m-%d-%H_%M')

    saved_loc = os.path.join('./content/VAE_Result/'+current_time)
    os.mkdir(saved_loc)

    print("?????? ??????: ", saved_loc)
    writer = SummaryWriter(saved_loc)
    EPOCHS = 20
    BATCH_SIZE = 32
    data, label, input_len = makeUserData()
    #pickleData.pickle_save(label, "./content/Embeddings/userlabel 200m.pkl")
    #pickleData.pickle_save(data,"./content/Embeddings/userdata.pkl")
    #data, label, input_len = makeCateData()
    #pickleData.pickle_save(data,"./content/Embeddings/catedata.pkl")
    # data = pickle_load("./content/POI(philadelphia)/normalizedUserVisitData.pkl")
    # label = pickle_load("./content/POI(philadelphia)/normalizedUserVisitData_label.pkl")
    np.random.shuffle(data)
    tr = data[:int(len(data)/10*8)]
    #tr_label = label[:int(len(label)/10*8)]
    te = data[int(len(data)/10*8):]
    #te_label = label[:int(len(label)/10*8)]
    trainset = CustomDataset(tr,tr)
    trainloader = DataLoader(trainset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 2)

    testset = CustomDataset(te,te)
    testloader = DataLoader(testset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 2)
    print(len(testset))
    print(len(trainset))
    
    VAE_model = VAE(input_len, 512, 256, 128).to(DEVICE)
    optimizer = optim.Adam(VAE_model.parameters(), lr = 1e-3)
    for epoch in tqdm(range(0, EPOCHS)):
        train(epoch, VAE_model, trainloader, optimizer, input_len)
        test(epoch, VAE_model, testloader,input_len)
        print("\n")
    getUserEmbedding(VAE_model)
    #pickleData.pickle_save(VAE_model,"./content/models/loss mean 200m cate8_2.pkl")
    writer.close()