import torchvision,torch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torchvision.utils as vutils

train_transforms=torchvision.transforms.Compose([torchvision.transforms.Resize((64,64)),torchvision.transforms.CenterCrop((64,64)),torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
train_data=torchvision.datasets.CIFAR100(root='./data',download=True,train=True,transform=train_transforms)
train_loader=torch.utils.data.DataLoader(train_data,shuffle=True,batch_size=32)

def weight_init(m):
    if isinstance(m,torch.nn.ConvTranspose2d) or isinstance(m,torch.nn.Conv2d):
        m.weight.data.normal_(0,0.02)
        m.bias.data.zero_()


class generator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net=torch.nn.Sequential(torch.nn.ConvTranspose2d(100,1024,4,1,0),torch.nn.BatchNorm2d(1024),torch.nn.ReLU(),torch.nn.ConvTranspose2d(1024,512,4,2,1),torch.nn.BatchNorm2d(512),torch.nn.ReLU(),torch.nn.ConvTranspose2d(512,256,4,2,1),torch.nn.BatchNorm2d(256),torch.nn.ReLU(),torch.nn.ConvTranspose2d(256,128,4,2,1),torch.nn.BatchNorm2d(128),torch.nn.ReLU(),torch.nn.ConvTranspose2d(128,3,4,2,1),torch.nn.Tanh())

    def weight_initialization(self):
        for i in self._modules:
            weight_init(self._modules[i])

    def forward(self,input):
        return self.net(input)

class discriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net=torch.nn.Sequential(torch.nn.Conv2d(3, 128, 4, 2, 1),torch.nn.BatchNorm2d(128),torch.nn.ReLU(),torch.nn.Conv2d(128,256, 4, 2, 1),torch.nn.BatchNorm2d(256),torch.nn.ReLU(),torch.nn.Conv2d(256, 512,4, 2, 1),torch.nn.BatchNorm2d(512),torch.nn.ReLU(),torch.nn.Conv2d(512,1024, 4, 2, 1),torch.nn.BatchNorm2d(1024),torch.nn.ReLU(),torch.nn.Conv2d(1024,1, 4, 1, 0),torch.nn.Sigmoid())

    def weight_initialization(self):
        for i in self._modules:
            weight_init(self._modules[i])

    def forward(self,input):
        return self.net(input)

device=torch.device('cuda')
gen=generator().to(device)
disc=discriminator().to(device)

gen.weight_initialization()
disc.weight_initialization()

def plot():
    img_list=[]
    z = torch.randn((32, 100)).view(-1, 100, 1,1).to(device)
    with torch.no_grad():
        test_images = gen(z).detach().cpu()
    img_list.append(vutils.make_grid(test_images,padding=2,normalize=True))
    plt.imshow(np.transpose(img_list[-1],(1,2,0)))
    plt.show()

    gen.train()

epochs=10
loss_criterion=torch.nn.BCELoss()
gen_optimizer=torch.optim.Adam(gen.parameters(),lr=0.0002,betas=(0.5,0.9))
disc_optimizer=torch.optim.Adam(disc.parameters(),lr=(0.0002/2),betas=(0.5,0.9))

for i in range(epochs):
    disc_loss=[]
    gen_loss=[]
    for img,_ in train_loader:
        disc.zero_grad()

        y_real=torch.ones(img.size()[0])
        y_fake=torch.zeros(img.size()[0])

        img,y_real,y_fake=img.to(device),y_real.to(device),y_fake.to(device)
        disc_op=disc(img).squeeze()
        d_real_loss=loss_criterion(disc_op,y_real)

        z=torch.randn((img.size()[0],100)).view(-1,100,1,1)
        z=z.to(device)
        gen_op=gen(z)

        disc_gen_op=disc(gen_op).squeeze()
        d_fake_loss=loss_criterion(disc_gen_op,y_fake)
        d_loss=d_real_loss+d_fake_loss

        d_loss.backward()
        disc_optimizer.step()
        disc_loss.append(d_loss.item())

        gen.zero_grad()
        z=torch.randn((img.size()[0],100)).view(-1,100,1,1)
        z=z.to(device)
        gen_op=gen(z)

        disc_gen_op=disc(gen_op).squeeze()
        g_loss=loss_criterion(disc_gen_op,y_real)
        g_loss.backward()
        gen_optimizer.step()
        gen_loss.append(g_loss.item())


    print("Epoch {}, d_loss:{:.2f}, g_loss:{:.2f}".format((i+1),torch.mean(torch.FloatTensor(disc_loss)),torch.mean(torch.FloatTensor(gen_loss))))
plot()
