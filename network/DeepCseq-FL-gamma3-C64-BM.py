import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SI364Dataset(Dataset): #rewrite Dataset to read own data
    def __init__(self, input_file, label_file):
        self.data = np.load(input_file)
        self.label = np.load(label_file)

    def __getitem__(self, index):
        return (torch.tensor(self.data[index],dtype=torch.float32),
                torch.tensor(self.label[index], dtype=torch.long)) #dtype=long for CEloss/Focal loss
        
    def __len__(self):
        return len(self.data)
    
class NetWork(nn.Module): 
    def __init__(self):
        super(NetWork,self).__init__()
        
        self.conv1 = nn.Conv2d(1, 128, (5,45), padding=(2,0))
        self.block1 = nn.ModuleList([
            nn.LayerNorm((600,1)), #self.block[0]
            nn.GLU(dim=1),#self.block[1]
            nn.Conv2d(64, 128, (5,1), padding=(2,0)) #self.block[2]
        ])
        for i in range(9):
            self.block1.append(nn.LayerNorm((600,1)))
            self.block1.append(nn.GLU(dim=1))
            self.block1.append(nn.Conv2d(64, 128, (5,1), padding=(2,0)))
        
        self.plain = nn.ModuleList([
            nn.LayerNorm((600,1)), #self.block[0]
            nn.GLU(dim=1),#self.block[1]
            nn.Conv2d(64, 128, (5,1), padding=(2,0)) #self.block[2]
        ])
        
        self.block2 = self.block1
        self.lastLN = nn.LayerNorm((600,1))
        self.lastGLU = nn.GLU(dim=1)
        self.lastconv1 = nn.Conv2d(64, 64, 1)
        self.lastconv2 = nn.Conv2d(64, 2, 1)
        
        pass

    def forward(self, x):
        x = self.conv1(x)
        for i in range(10): #basic block * 
            x1 = x
            x = self.block2[3*i](x)
            x = self.block2[3*i+1](x)
            x = self.block2[3*i+2](x)
            x = x + x1
        
        for i in range(3):
            x = self.plain[i](x)
            
        for i in range(10):
            x1 = x
            x = self.block1[3*i](x)
            x = self.block1[3*i+1](x)
            x = self.block1[3*i+2](x)
            x = x + x1
            
        x = self.lastLN(x)
        x = self.lastGLU(x)
        x = F.relu(self.lastconv1(x))
        x = F.softmax(self.lastconv2(x), dim=1)
        output2 = torch.split(x, [1,1], dim=1)
        x = torch.cat([output2[0].squeeze(dim=1),output2[1].squeeze(dim=1)],-1)
        # x = x.permute(0, 2, 1)
        return x

class focal_loss(nn.Module):    
    def __init__(self, alpha=0.25, gamma=2, num_classes = 3, size_average=True):
        """
        focal_loss????????????, -??(1-yi)**?? *ce_loss(xi,yi)      
        ???????????????????????? focal_loss????????????.
        :param alpha:   ???????????,????????????.      ?????????????????,??????????????????,?????????????????,???????????????[??, 1-??, 1-??, ....],????????? ???????????????????????????????????? , retainnet????????????0.25
        :param gamma:   ????????,????????????????????????. retainnet????????????2
        :param num_classes:     ????????????
        :param size_average:    ??????????????????,???????????????
        """

        super(focal_loss,self).__init__()
        self.size_average = size_average
        if isinstance(alpha,list):
            assert len(alpha)==num_classes   # ???????????list????????????,size:[num_classes] ??????????????????????????????????????????
            # print("Focal_loss alpha = {}, ??????????????????????????????????????????".format(alpha))
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha<1   #???????????????????????,???????????????????????????,??????????????????????????????
            # print(" --- Focal_loss alpha = {} ,???????????????????????????,????????????????????????????????? --- ".format(alpha))
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1-alpha) # ?? ????????? [ ??, 1-??, 1-??, 1-??, 1-??, ...] size:[num_classes]
        self.gamma = gamma

    def forward(self, preds, labels):
        """
        focal_loss????????????        
        :param preds:   ????????????. size:[B,N,C] or [B,C]    ????????????????????????????????????, B ??????, N????????????, C?????????        
        :param labels:  ????????????. size:[B,N] or [B]        
        :return:
        """        
        # assert preds.dim()==2 and labels.dim()==1        
        preds = preds.contiguous().view(-1,preds.size(-1))        
        # print(preds.shape)
        self.alpha = self.alpha.to(preds.device)        
        preds_softmax = F.softmax(preds, dim=1) # ???????????????????????????log_softmax, ?????????????????????softmax?????????(????????????????????????log_softmax,????????????exp??????)        
        preds_logsoft = torch.log(preds_softmax)
        preds_softmax = preds_softmax.gather(1,labels.view(-1,1))   # ???????????????nll_loss ( crossempty = log_softmax + nll )        
        preds_logsoft = preds_logsoft.gather(1,labels.view(-1,1))        
        self.alpha = self.alpha.gather(0,labels.view(-1))        
        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) ???focal loss??? (1-pt)**??
        loss = torch.mul(self.alpha, loss.t())        
        if self.size_average:        
            loss = loss.mean()        
        else:            
            loss = loss.sum()        
        return loss

def train_loop(dataloader, model):
    loss_fn = focal_loss(alpha=[1,5], gamma=3, num_classes=2)
    optimizer = optim.Adam(model.parameters(),lr=0.001)
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X.to(device=device))
        y = y.to(device=device)
        # print(y.shape)
        # print(pred.shape)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch%100 == 0:
            loss, current = loss, batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5f}/{size:>5f}]")
    
def test_loop(dataloader, model):
    loss_fn = nn.CrossEntropyLoss()
    num_batches = len(dataloader)
    model.eval()
    
    # acc = torch.zeros(1).to(device)
    with torch.no_grad():
        fp, tp, fn, tn = 0, 0, 0, 0
        total_num = 0
        for X, y in dataloader:
            pred = model(X.to(device=device)) # [600 (batch, 2)]
            y = y.to(device=device)
            # print(pred.shape)
            # print(y.shape)
            # acc = metric(pred.to(device), y.to(device)
            for batch_size in range(20):
                for res_length in range(600):
                    try:
                        if torch.argmax(pred[batch_size,res_length,:]) == 1:
                            if y[batch_size,res_length] == 1:
                                tp += 1
                            elif y[batch_size,res_length] == 0:
                                fp += 1
                        elif torch.argmax(pred[batch_size,res_length,:]) == 0:
                            if y[batch_size,res_length] == 0:
                                tn += 1
                            elif y[batch_size,res_length] == 1:
                                fn += 1
                        total_num += 1
                    except:
                        pass
                    # acc += torch.sum(torch.argmax(pred[:,:,i][j]) == y[:,i][j])#
            # acc = acc/ (600*len(dataloader))#
            # acc = count / 12000
        acc = (tp+tn) / total_num
        recall = tp / (tp+fn)
        precision = tp / (tp+fp)
        MCC1 = (tp * tn) - (fp * fn)
        MCC2 = ((tp + fp)*(tp + fn)*(tn + fp)*(tn + fn))**0.5
        MCC = MCC1 / MCC2

    network.train()
    print(f"Test Matrix: \n TP: {tp} \n TN: {tn} \n FN: {fn} \n FP: {fp} \n")
    print(f"Test Error: \n Accuracy: {acc*100:>0.1f}%\n Recall: {recall*100:>0.1f}%\n Precision: {precision*100:>0.1f}%\n MCC: {MCC:>0.001f}\n")
    

if __name__ == '__main__':
    train_data = SI364Dataset('data/total_train_feature_BM.npy', 'data/total_train_label_BM.npy')
    train_dataloader = DataLoader(train_data, batch_size=60, shuffle=True)

    test_data = SI364Dataset('data/total_test_feature_BM.npy', 'data/total_test_label_BM.npy')
    test_dataloader = DataLoader(test_data, batch_size=60, shuffle=False)

    network = NetWork()
    network.to(device)

    for epoch in range(20):
        print('Epoch: %s' %str(epoch+1))
        train_loop(train_dataloader, network)
        test_loop(test_dataloader, network)
