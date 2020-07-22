import numpy as np
#import pandas as pd 
import matplotlib.pyplot as plt
import datetime as dt
import torch
import sys
sys.path.append("../tools/")
from projeto import preprocess
from torchsummary import summary
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix

class MLP(nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        self.hidden1 = nn.Linear(22,50)
        self.hidden2 = nn.Linear(50,25)
        self.hidden3 = nn.Linear(25,10)
        self.output = nn.Linear(10,1)

        self.sigmoid = nn.Sigmoid()
        # self.softmax = nn.Softmax()
        self.relu = nn.ReLU()
    
    def forward(self,x):
        x = self.hidden1(x)
        x = self.relu(x)
        x = self.hidden2(x)
        x = self.relu(x)
        x = self.hidden3(x)
        x = self.relu(x)
        x = self.output(x)
        #x = self.relu(x)
       # x = self.softmax(x)
        x = self.sigmoid(x)
        return x

class covid_dataset(torch.utils.data.Dataset):
    def __init__(self,values,labels):
        self.val = values
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):

        x = self.val[index]
        y = self.labels[index]

        return x,y


def get_weights(y):
    weights = []
    wg = np.count_nonzero(y==0)/np.count_nonzero(y==1)
    wg = 1
    print("Setting weight to {}".format(wg))
    for i in y:
        if i == 1:
            weights.append(wg)
        else:
            weights.append(1)
    return torch.FloatTensor(weights)

def run(num_workers=0,batch_size=64):
    torch.cuda.empty_cache()
    X_train, X_test, Y_train, Y_test, data, target = preprocess()
    print(X_test.shape)
    X_train = X_train.astype(float).to_numpy().reshape(-1,22)
    Y_train = Y_train.astype(int).to_numpy().reshape(-1,1)


    # X_test = X_test.astype(float).to_numpy().reshape(-1,22)
    # Y_test = Y_test.astype(int).to_numpy().reshape(-1,1)

    X_test, X_val, Y_test, Y_val = train_test_split(X_test,Y_test,train_size=0.5)

    X_test = X_test.astype(float).to_numpy().reshape(-1,22)
    Y_test = Y_test.astype(int).to_numpy().reshape(-1,1)

    X_val = X_val.astype(float).to_numpy().reshape(-1,22)
    Y_val = Y_val.astype(int).to_numpy().reshape(-1,1)

    
    print(X_train.shape)
    print(Y_train.shape)
    

    cov_set = covid_dataset(X_train,Y_train)
    val_set = covid_dataset(X_val,Y_val)
    weights = get_weights(Y_train)
    sampler = WeightedRandomSampler(weights,len(weights))
    #data_generator = DataLoader(cov_set,batch_size=batch_size,num_workers=num_workers,sampler=sampler)
    data_generator = DataLoader(cov_set,batch_size=batch_size,num_workers=num_workers)
    val_generator = DataLoader(val_set,batch_size=batch_size,num_workers=num_workers)

    mlp = MLP()
    mlp.cuda()
    summary(mlp,(1,22))
    loss_fn = nn.BCELoss()
    loss_fn.cuda()
    #optimizer = torch.optim.SGD(mlp.parameters(),lr=0.001,momentum=0.1)
    optimizer = torch.optim.AdamW(mlp.parameters(), lr=0.001)

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=5)
    
    epochs = 100
    input("Press any key to start\n")
    for t in range(epochs):
        print("Epoch = {}".format(t))
        for x_train, y_train in data_generator:
            x_train, y_train = x_train.cuda(), y_train.cuda()
            y_pred = mlp(x_train.float())
            loss = loss_fn(y_pred,y_train.float())
            print("Loss = {}".format(loss.item()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            val_loss = 0
            n = 0
            for x_val, y_val in val_generator:
                x_val, y_val = x_val.cuda(), y_val.cuda()
                y_pred_val = mlp(x_val.float())
                val_loss += loss_fn(y_pred_val,y_val.float())
                n +=1
            val_loss /=n
            print("Val Loss = {}".format(val_loss.item()))
            lr_scheduler.step(val_loss)  


    X_test = torch.from_numpy(X_test).cuda()

    y_pred = mlp(X_test.float()).cpu().detach().numpy()

    preds = y_pred > 0.5

    acc_scor = accuracy_score(Y_test,preds)
    prec_scor = precision_score(Y_test,preds)
    f1_scor = f1_score(Y_test,preds)
    rec_scor = recall_score(Y_test,preds)
    tn, fp, fn, tp = confusion_matrix(Y_test, preds).ravel()
    print(np.count_nonzero(Y_test==0))
    print(np.count_nonzero(Y_test==1))
    print("Accuracy: {}".format(acc_scor) + "     Precision: {}".format(prec_scor) + "     Recall: {}".format(rec_scor) + "     F1: {}".format(f1_scor))
    print("TN: {}".format(tn)+"     FP: {}".format(fp)+"     FN: {}".format(fn)+"      TP: {}".format(tp))
    plt.figure()
    plt.stem(list(range(0,150)),Y_test[0:150],'g', markerfmt='go')
    plt.stem(list(range(0,150)),preds[0:150],'--r', markerfmt='ro')
    plt.figure()
    plt.plot(y_pred)
    plt.show()
    if input("Save file (s) ?\n") == 's':
        torch.save(MLP,"mlp_model.model")
if __name__=="__main__":
    run()