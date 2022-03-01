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
parser.add_argument('--dropout_rate',dest='dropout_rate',default=0.0,type=float,help="Dropout Rate")
parser.add_argument('--lr',dest='lr',default=1e-2,type=float,help="Learning Rate")
parser.add_argument('--scratch',dest='scratch',default='False',nargs='?',type=str2bool,help="Train from Scratch?")
parser.add_argument('--explanation',dest='explanation',default='',type=str,help="Explanation (If necessary)")
parser.add_argument('--seed',dest='seed',default=10,type=int,help="Random Seed")
parser.add_argument('--normalise',dest='normalise',default='standard',type=str,help="Normalisation Type (standard/const)")
parser.add_argument('--save',dest='save',default=False,type=str2bool,nargs='?',help="Save Model ? ")

args = parser.parse_args()
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

name = 'Expt{}_l{}_ep{}_early{}_reg{}_dr{}_rate_{}_act{}_hid{}_lr{}_sc{}_bch{}__norm{}_seed{}'.format(args.expt,args.layers,args.epochs,args.early,args.reg,args.dropout,args.dropout_rate,args.activation,args.hid,args.lr,args.scratch,args.batch,args.normalise,args.seed)
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

if args.expt==1:
    data_train,data_valid,data_test = load_mnist(args)
    model = Lin_Net(args).to(args.device)
elif args.expt==2:
    data_train,data_valid,data_test = load_cifar(args)
    model = Res_Net(args).to(args.device)
elif args.expt==3:
    data_train,data_valid,data_test = load_tinycifar(args)
    model = Res_Net(args).to(args.device)
else:
    raise ValueError



param_to_update = []
for param in model.parameters():
    if param.requires_grad==True:
        param_to_update.append(param)

print("Number of parameters : ",len(param_to_update))
if args.reg:
    optimizer = optim.Adam(params=param_to_update,lr=args.lr,weight_decay=1e-3)
else:
    optimizer = optim.SGD(params = param_to_update,lr=args.lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=5,min_lr=1e-5)




train(model,data_train,data_valid,optimizer,scheduler,args)