import torch.optim as optim
import torch
import os
from argparse import ArgumentParser,ArgumentTypeError
from utils import *

import warnings
warnings.filterwarnings("ignore")

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')


parser = ArgumentParser()

# Arguments

parser.add_argument('--expt',dest='expt',default=1,type=int,help="Expt Number")
parser.add_argument('--layers',dest='layers',default=3,type=int,help="Hidden Layers")
parser.add_argument('--activation',dest='activation',default='relu',type=str,help="Activation Function")
parser.add_argument('--epochs',dest='epochs',default=5,type=int,help="Number of Epochs")
parser.add_argument('--early',dest='early',default=0,type=int,help="Early Stopping Patience (0) for no ES")
parser.add_argument('--batch',dest='batch',default=16,type=int,help="Batch Size")
parser.add_argument('--hid',dest='hid',default=500,type=int,help="Hidden Dimensions")
parser.add_argument('--reg',dest='reg',default='False',const=True,nargs='?',type=str2bool,help="Regularisation")
parser.add_argument('--dropout',dest='dropout',default='False',const=True,nargs='?',type=str2bool,help="Dropout")
parser.add_argument('--lr',dest='lr',default=1e-2,type=float,help="Learning Rate")
parser.add_argument('--scratch',dest='scratch',default='True',nargs='?',type=str2bool,help="Train from Scratch?")
parser.add_argument('--explanation',dest='explanation',default='',type=str,help="Explanation (If necessary)")
parser.add_argument('--seed',dest='seed',default=10,type=int,help="Random Seed")
parser.add_argument('--normalisation',dest='normalise',default='standard',type=str,help="Normalisation Type (standard/const)")

args = parser.parse_args()
args.device = 'gpu' if torch.cuda.is_available() else 'cpu'

name = 'Expt{}_l{}_ep{}_early{}_reg{}_dr{}_act{}_hid{}_lr{}_sc{}_bch{}__norm{}_seed{}'.format(args.expt,args.layers,args.epochs,args.early,args.reg,args.dropout,args.activation,args.hid,args.lr,args.scratch,args.batch,args.normalise,args.seed)
args.name = name

try:
    os.mkdir("Results")
except:
    pass

try:
    os.mkdir("Results/"+name)
except:
    pass

torch.manual_seed(args.seed)
data_train,data_valid,data_test = load_mnist(args)
model = Lin_Net(args).to(args.device)
optimizer = optim.Adam(params=model.parameters(),lr=args.lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=5,min_lr=1e-5)


def train(model,data_train,data_valid,optimizer,scheduler,args):
    losses = []
    train_accuracy = []
    for epoch in range(0,args.epochs):
        epoch_loss = 0
        correct = 0
        loss = torch.tensor([0],dtype=torch.float32)
        print("Epoch Number : ",epoch)
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
            

        _,acc = test(model,data_valid)
        scheduler.step(acc)
        train_accuracy.append(correct/len(data_train))
        losses.append(epoch_loss/len(data_train))
        
    print(losses,train_accuracy)

def test(model,data):
    test_loss = 0
    test_accuracy = 0
    with torch.no_grad():
        correct = 0
        for x,y in data:
            x = x.to(args.device)
            y = y.to(args.device)
            pred = model(x).type(torch.float32)
            loss = torch.nn.functional.nll_loss(pred,y)
            test_loss += loss.item()*len(x)
            _,output = torch.max(pred,dim=1)
            correct += (output == y).detach().float().sum().item()
            break

        test_accuracy = correct/len(data)
        
    
    return test_loss, test_accuracy


train(model,data_train,data_valid,optimizer,scheduler,args)