import torch
from torchvision.datasets import MNIST,CIFAR10
from torchvision.transforms import Compose,ToTensor,Normalize
from torch.utils.data import random_split
import torch.nn as nn
import torch.nn.functional as F


def load_mnist(args):
    if args.normalise=='standard':
        mean,std = (0.1307,), (0.3081,)
    elif args.normalise=='constant':
        mean,std = (0,),(1/255.0,)
    else:
        raise ValueError


    transforms = Compose([ToTensor(),Normalize(mean,std)])
    data = MNIST(root="mnist/",train=True,download=True,transform=transforms)
    test = MNIST(root="mnist/",train=False,transform=transforms)
    train,val = random_split(data,[int(0.70*len(data)),int(0.30*len(data))],generator=torch.Generator().manual_seed(args.seed))

    print("Train : {}, Validation : {}, Test : {} ".format(len(train),len(val),len(test)))

    train = torch.utils.data.DataLoader(train,batch_size=args.batch,shuffle=True)    
    val = torch.utils.data.DataLoader(val,batch_size=args.batch,shuffle=True)    
    test = torch.utils.data.DataLoader(test,batch_size=32,shuffle=True)  
    for x,y in train:
        print(type(x),type(y),x.shape,y.shape)
        break

    return train,val,test

def load_cifar():
    pass

def load_tinycifar():
    pass

def train(dataloader,model,args):
    pass

def test(dataloader,model,args):
    pass


class Lin_Net(nn.Module):
    def __init__(self,args):
        super(Lin_Net,self).__init__()
        self.args = args
        self.fcs = nn.ModuleList()
        self.flatten = nn.Flatten()
        if args.activation == 'relu':
            self.activation = F.relu
        elif args.activation == 'sigmoid':
            self.activation = F.sigmoid
        else:
            self.activation = F.tanh                           

        for i in range(0,args.layers-1):
            if i==0:
                self.fcs.append(nn.Linear(784,args.hid))
            else:
                self.fcs.append(nn.Linear(args.hid,args.hid))
        
        self.fcs.append(nn.Linear(args.hid,10))

    def forward(self,x):
        x = self.flatten(x)
        for fc in self.fcs[:-1]:
            x = fc(x)
            x = self.activation(x)

        if self.args.dropout:
            x = F.dropout(x,0.2,training=self.training)
        x = self.fcs[-1](x)
        x = F.log_softmax(x)

        return x
        