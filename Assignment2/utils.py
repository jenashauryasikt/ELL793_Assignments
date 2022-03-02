import torch
import torchvision.models as models
from torchvision.datasets import MNIST,CIFAR10
from torchvision.transforms import Compose,ToTensor,Normalize,RandomHorizontalFlip,RandomRotation, RandomVerticalFlip, RandomApply
from torch.utils.data import random_split,Dataset
import torch.nn as nn
import torch.nn.functional as F
import random

class TinyCifar(Dataset):
    def __init__(self,args,train):
        self.train = train
        random.seed(args.seed)
        if args.normalise=='standard':
            mean,std = [0.485, 0.456, 0.406],[0.229, 0.224, 0.225]
        elif args.normalise=='constant':
            mean,std = (0,),(255.0,)
        else:
            raise ValueError
        
        self.test_transforms = Compose([ToTensor(),Normalize(mean,std)])
        self.train_transforms =  Compose([ToTensor(),Normalize(mean,std)])
        self.augment_transforms = Compose([RandomHorizontalFlip(),RandomVerticalFlip(),RandomRotation(-10,10)])

        data = CIFAR10(root="cifar/",train=True,download=True,transform=self.train_transforms)
        tiny_data = {}
        self.new_data = []
        if self.train:
            for label in range(0,10):
                tiny_data[label] = [x for x in data if x[1]==label]
            
            print(len(tiny_data[0]))
            for label in range(0,10):
                self.new_data.extend(random.sample(tiny_data[label],500))

            random.shuffle(self.new_data)
            assert len(self.new_data)==5000
        else:
            self.test = CIFAR10(root="cifar/",train=False,transform=self.test_transforms)


    def __getitem__(self, index):
        if self.train:
            img,y = self.new_data[index]
            img = self.augment_transforms(img)
            return (img,y)
        else:
            img,y = self.test[index]
            return (img,y)
    
    def __len__(self):
        if self.train:
            return 5000
        else:
            return 10000
        


def load_mnist(args):
    if args.normalise=='standard':
        mean,std = (0.1307,), (0.3081,)
    elif args.normalise=='constant':
        mean,std = (0,),(255.0,)
    else:
        raise ValueError


    transforms = Compose([ToTensor(),Normalize(mean,std)])
    data = MNIST(root="mnist/",train=True,download=True,transform=transforms)
    test = MNIST(root="mnist/",train=False,transform=transforms)
    train,val = random_split(data,[50000,10000],generator=torch.Generator().manual_seed(args.seed))

    print("Train : {}, Validation : {}, Test : {} ".format(len(train),len(val),len(test)))

    train = torch.utils.data.DataLoader(train,batch_size=args.batch,shuffle=True)    
    val = torch.utils.data.DataLoader(val,batch_size=args.batch,shuffle=True)    
    test = torch.utils.data.DataLoader(test,batch_size=32,shuffle=True)  
    for x,y in train:
        print(type(x),type(y),x.shape,y.shape)
        break

    return train,val,test

def load_cifar(args):
    if args.normalise=='standard':
        mean,std = [0.485, 0.456, 0.406],[0.229, 0.224, 0.225]
    elif args.normalise=='constant':
        mean,std = (0,),(255.0,)
    else:
        raise ValueError


    test_transforms = Compose([ToTensor(),Normalize(mean,std)])
    train_transforms =  Compose([ToTensor(),Normalize(mean,std)])

    data = CIFAR10(root="cifar/",train=True,download=True,transform=train_transforms)
    test = CIFAR10(root="cifar/",train=False,transform=test_transforms)
    train,val = random_split(data,[42000,8000],generator=torch.Generator().manual_seed(args.seed))

    print("Train : {}, Validation : {}, Test : {} ".format(len(train),len(val),len(test)))

    train = torch.utils.data.DataLoader(train,batch_size=args.batch,shuffle=True)    
    val = torch.utils.data.DataLoader(val,batch_size=args.batch,shuffle=True)    
    test = torch.utils.data.DataLoader(test,batch_size=32,shuffle=True)  
    for x,y in train:
        print(type(x),type(y),x.shape,y.shape)
        break

    return train,val,test

    pass

def load_tinycifar(args):
    data = TinyCifar(args,train=True)
    test = TinyCifar(args,train=False)
    train,val = random_split(data,[4200,800],generator=torch.Generator().manual_seed(args.seed))

    print("Train : {}, Validation : {}, Test : {} ".format(len(train),len(val),len(test)))

    train = torch.utils.data.DataLoader(train,batch_size=args.batch,shuffle=True)    
    val = torch.utils.data.DataLoader(val,batch_size=args.batch,shuffle=True)    
    test = torch.utils.data.DataLoader(test,batch_size=32,shuffle=True)  
    for x,y in train:
        print(type(x),type(y),x.shape,y.shape)
        break

    return train,val,test

   

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
            x = F.dropout(x,self.args.dropout_rate,training=self.training)
        x = self.fcs[-1](x)
        x = F.log_softmax(x,dim=1)

        return x


class Res_Net(nn.Module):
    def __init__(self,args):
        super(Res_Net,self).__init__()
        self.args = args
        self.backbone = models.resnet18(pretrained=True)
        self.fcs = nn.ModuleList()
        self.flatten = nn.Flatten()
        
        if not args.scratch:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        if args.activation == 'relu':
            self.activation = F.relu
        elif args.activation == 'sigmoid':
            self.activation = F.sigmoid
        else:
            self.activation = F.tanh                           

        for i in range(0,args.layers-1):
            if i==0:
                self.fcs.append(nn.Linear(self.backbone.fc.out_features,args.hid))
            else:
                self.fcs.append(nn.Linear(args.hid,args.hid))
        
        self.fcs.append(nn.Linear(args.hid,10))

    def forward(self,x):
        x = self.backbone(x)
        #print(x.shape,self.backbone.fc.out_features)
        for fc in self.fcs[:-1]:
            x = fc(x)
            x = self.activation(x)

        if self.args.dropout:
            x = F.dropout(x,self.args.dropout_rate,training=self.training)
        x = self.fcs[-1](x)
        x = F.log_softmax(x,dim=1)

        return x


def train(model,data_train,data_valid,optimizer,scheduler,args):
    losses = []
    train_accuracy = []
    patience = args.early if args.early > 0 else -1
    no_improvement = 0
    best = 0
    for epoch in range(0,args.epochs):
        epoch_loss = 0
        correct = 0
        total = 0
        for x,y in data_train:
            x = x.to(args.device)
            y = y.to(args.device)
            optimizer.zero_grad()
            pred = model(x).type(torch.float32)
            loss = torch.nn.functional.nll_loss(pred,y)
            loss.backward()
            epoch_loss += loss.item()*len(x)
            optimizer.step()
            _,output = torch.max(pred,dim=1)
            correct += (output == y).detach().float().sum().item()
            total = total + x.shape[0]

        val_loss,val_acc = test(model,data_valid,args)
        train_loss = epoch_loss/total
        train_acc = correct/total

        print("Epoch : {} ,Train Acc : {}, Train Loss : {}, Val Acc : {}, Val Loss : {}".format(epoch,train_acc,train_loss,val_acc,val_loss))
        
        with open("Results/{}/log.txt".format(args.name),'a') as f:
           f.write("Epoch : {} ,Train Acc : {}, Train Loss : {}, Val Acc : {}, Val Loss : {}\n".format(epoch,train_acc,train_loss,val_acc,val_loss))
        
        if val_acc > best + 0.01:
            best = val_acc
            no_improvement = 0
            if args.save:
                torch.save(model.state_dict(),"Results/{}/model_best.pth".format(args.name))

        if args.save:
            torch.save(model.state_dict(),"Results/{}/model_{}.pth".format(args.name,epoch))

        no_improvement += 1

        if patience > 0 and no_improvement == patience:
            break

        scheduler.step(val_acc)
        train_accuracy.append(train_acc)
        losses.append(train_loss)
        
    return (losses,train_accuracy)

def test(model,data,args):
    test_loss = 0
    test_accuracy = 0
    with torch.no_grad():
        correct = 0
        total = 0
        for x,y in data:
            x = x.to(args.device)
            y = y.to(args.device)
            pred = model(x).type(torch.float32)
            loss = torch.nn.functional.nll_loss(pred,y)
            _,output = torch.max(pred,dim=1)
            correct += (output == y).detach().float().sum().item()
            total = total + x.shape[0]

        test_accuracy = correct/total
        test_loss = loss/total
        
    
    return test_loss, test_accuracy

if __name__ == '__main__':
    class temp:
        def __init__(self):
            self.normalise = 'standard'
            self.seed = 10
    args = temp() 
    data = TinyCifar(args,True)
    print(type(data),len(data),data[0])