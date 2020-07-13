# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 08:41:30 2020

@author: Acioli
"""


import pandas as pd
from sklearn.model_selection import train_test_split
import datetime as dt

def preprocess(file_name='../tools/covid19-al-sintomas.csv'):

    covid = pd.read_csv(file_name)
    
    #lista de indices a serem removidos 
    index_to_remove = []
    #tempo de isolamento domiciliar
    isolation_days_range = 14
    current_year = 2020
    #inicio da pandamia, os primeiros casos foram registrados a partir desta data
    pandemic_start_month = 3
    
    # Descarta os casos de 'isolamento domiciliar' recentes (menos de 14 dias),
    # também descarta casos com mes ou ano marcado incorretamente
    for index, row in covid.iterrows():
        # data do atual
        today = dt.date.today()
        # data do atendimento 
        date = dt.datetime.strptime(row['data_atendimento'], 
                                    "%Y-%m-%dT%H:%M:%S.%fZ").date()
        # diferença de dias entre as datas
        delta = today - date
        # flag sobre a instancia se tratar ou não de isolamento domiciliar
        is_isolation = row['situacao_atual'] == 'Isolamento Domiciliar'
        
        # Instancias com ano incorreto
        if date.year > current_year:
            index_to_remove.append(index)
        # Instancias com meses incorretos
        elif date.month < pandemic_start_month:
            index_to_remove.append(index)
        # Instancias que tem menos de 14 dias e representa isolamento domiciliar
        elif is_isolation & (delta.days < isolation_days_range):
            index_to_remove.append(index)
    
    covid.drop(covid.index[index_to_remove], inplace=True )
    
    # exclusão de colunas inicialmente não relevantes
    covid = covid.drop(['id', 'etnia', 'municipio_residencia',
                        'classificacao', 'data_resultado_exame',
                        'data_atendimento', 'tipo_coleta', 
                        'data_obito', 'data_confirmacao_obito',
                        'idoso', 'profissional_saude',
                        'outros', 'outros_fatores'], axis=1)
    
    # exclusão de colunas sem marcação
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
    # covid = covid[covid['situacao_atual'] != 'Isolamento Domiciliar']
    
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
    situacao = covid.pop('situacao_atual')
    
    # print(covid.info())
    
    #Dividir o conjunto de informações em dados de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(covid, situacao, 
                                                        test_size=0.2, 
                                                        random_state=42)
    
    return X_train, X_test, y_train, y_test

