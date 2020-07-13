import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

class MLP(nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        self.hidden1 = nn.Linear(23,50)
        self.hidden2 = nn.Linear(50,25)
        self.hidden3 = nn.Linear(25,10)
        self.output = nn.Linear(10,1)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
    
    def forward(self,x):
        x = self.hidden1(x)
        x = self.relu(x)
        x = self.hidden2(x)
        x = self.relu(x)
        x = self.hidden3(x)
        x = self.relu(x)
        x = self.output(x)
        x = self.sigmoid(x)
        return x

class covid_datset(torch.utils.data.Dataset):
    def __init__(self,values,labels):
        self.val = values
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):

        x = self.val[index]
        y = self.labels[index]

        return x,y

def pre_processing(data):
    covid = pd.read_csv(data)

    #exclusão de colunas inicialmente não relevantes
    covid = covid.drop(['id', 'etnia', 'municipio_residencia',
                        'classificacao', 'data_resultado_exame',
                        'data_atendimento', 'tipo_coleta', 
                        'data_obito', 'data_confirmacao_obito',
                        'idoso', 'profissional_saude',
                        'outros', 'outros_fatores'], axis=1)

    #exclusão de colunas sem marcação
    covid = covid.drop(['ausegia', 'anosmia', 'nausea_vomito',
                        'coriza', 'congestao_nasal', 'calafrio'], axis=1)

    #exclusão das instancia cujo fator não foi informado
    covid = covid[covid['fator_nao_informado'] != 'X']

    #exclusão da coluna fator_nao_informado
    covid = covid.drop(['fator_nao_informado'], axis=1)

    #exclusão das instancia cujos sintomas não foram informados
    covid = covid[covid['nao_informado'] != 'X']

    #exclusão da coluna nao_informado
    covid = covid.drop(['nao_informado'], axis=1)

    #exclusão das instâncias de casos que ainda estão em aberto

    #pacientes que estão em isolamento domiciliar
    covid = covid[covid['situacao_atual'] != 'Isolamento Domiciliar']

    #pacientes que estão em internação UTI
    covid = covid[covid['situacao_atual'] != 'Internação UTI']

    #pacientes que estão em internamento de leite clínico
    covid = covid[covid['situacao_atual'] != 'Internação Leito Clínico']

    #Substituição de valores

    #Substitue os valores NaN por 0 (zero)
    covid = covid.fillna(0)
    #Substitue os valores X por 1 
    covid = covid.replace('X', 1)
    #Substitue valores Masculino e Feminino
    covid = covid.replace('Masculino', 1)
    covid = covid.replace('Mascuino', 1)
    covid = covid.replace('Feminino', 0)

    #Substitue valores na coluna situacao_atual
    covid = covid.replace('Óbito', 1)
    #Substitue todas as outras situacoes em 0 (não morte)
    covid = covid.replace(to_replace='[a-zA-Z]', value=0, regex=True)

    #exclusão das instâncias que não têm nenhum valor
    covid = covid.drop(covid[(covid['febre'] == 0) 
                            & (covid['tosse'] == 0)
                            & (covid['cefaleia'] == 0)
                            & (covid['dificuldade_respiratoria'] == 0)
                            & (covid['dispineia'] == 0)
                            & (covid['mialgia'] == 0)
                            & (covid['saturacao_menor_noventa_cinco'] == 0)
                            & (covid['adinofagia'] == 0)
                            & (covid['diarreia'] == 0)
                            & (covid['adinamia'] == 0)
                            & (covid['doenca_cardiovascular'] == 0)
                            & (covid['diabetes'] == 0)
                            & (covid['doenca_respiratoria_cronica'] == 0)
                            & (covid['hipertensao'] == 0)
                            & (covid['paciente_oncologico'] == 0)
                            & (covid['obesidade'] == 0)
                            & (covid['doenca_renal_cronica'] == 0)
                            & (covid['doenca_auto_imune'] == 0)
                            & (covid['asma'] == 0)
                            & (covid['sem_comorbidade'] == 0)
                            & (covid['pneumopatia'] == 0)].index)


    #mover coluna situação_atual para o final do dataframe
    y = covid.pop('situacao_atual')
    #Normalizando idade
    x = covid[['idade']].values.astype(float)
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    covid['idade'] = x_scaled

    return covid.astype(float).to_numpy(),y.astype(int).to_numpy()

def get_weights(y):
    weights = []
    for i in y:
        if i == 1:
            weights.append(4)
        else:
            weights.append(1)
    return torch.FloatTensor(weights)

def run(num_workers=0,batch_size=200):
    f = "covid19-al-sintomas.csv"
    torch.cuda.empty_cache()
    data,y = pre_processing(f)
    data = data.reshape((-1,23))
    y = y.reshape((-1,1))
    #data = data.to_numpy()

    X_train,X_test,Y_train,Y_test = train_test_split(data,y,test_size=0.05,shuffle=True)
    print(X_train.shape)
    print(Y_train.shape)
    cov_set = covid_datset(X_train,Y_train)
    weights = get_weights(Y_train)
    sampler = WeightedRandomSampler(weights,len(weights))
    data_generator = DataLoader(cov_set,batch_size=batch_size,num_workers=num_workers,sampler=sampler)

    mlp = MLP()
    mlp.cuda()
    loss_fn = nn.BCEWithLogitsLoss()
    loss_fn.cuda()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=0.001)
    epochs = 10
    for t in range(epochs):
        print("Epoch = {}".format(t))
        for x_train, y_train in data_generator:
            x_train, y_train = x_train.cuda(), y_train.cuda()
            y_pred = mlp(x_train.float())
            y_pred = y_pred.reshape((-1,1))
            y_train = y_train.reshape((-1,1))
            loss = loss_fn(y_pred,y_train.float())
            print("Loss = {}".format(loss.item()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    X_test = torch.from_numpy(X_test).cuda()

    y_pred = mlp(X_test.float()).cpu().detach().numpy()

    preds = []
    for pr in y_pred:
        if pr > 0.5:
            preds.append(1)
        else:
            preds.append(0)

    acc_scor = accuracy_score(Y_test,preds)
    prec_scor = precision_score(Y_test,preds)
    f1_scor = f1_score(Y_test,preds)
    rec_scor = recall_score(Y_test,preds)
    print("Accuracy: {}".format(acc_scor) + "     Precision: {}".format(prec_scor) + "     Recall: {}".format(rec_scor) + "     F1: {}".format(f1_scor))

    plt.figure()
    plt.plot(Y_test,'b')
    plt.plot(preds,'r')
    plt.show()


if __name__=="__main__":
    run()