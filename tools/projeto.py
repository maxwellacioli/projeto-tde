# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 08:41:30 2020

@author: Acioli
"""


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import datetime as dt

def preprocess(file_name='../tools/covid19-al-sintomas.csv'):

    covid_df = pd.read_csv(file_name)
    
    #lista de indices a serem removidos 
    index_to_remove = []
    #tempo de isolamento domiciliar
    isolation_days_range = 14
    current_year = 2020
    #inicio da pandamia, os primeiros casos foram registrados a partir desta data
    pandemic_start_month = 3
    
    # Descarta os casos de 'isolamento domiciliar' recentes (menos de 14 dias),
    # também descarta casos com mes ou ano marcado incorretamente
    for index, row in covid_df.iterrows():
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
    
    covid_df.drop(covid_df.index[index_to_remove], inplace=True )
    
    # exclusão de colunas inicialmente não relevantes
    covid_df = covid_df.drop(['id', 'etnia', 'municipio_residencia',
                        'classificacao', 'data_resultado_exame',
                        'data_atendimento', 'tipo_coleta', 
                        'data_obito', 'data_confirmacao_obito',
                        'idoso', 'profissional_saude',
                        'outros', 'outros_fatores', 'doenca_auto_imune'], axis=1)
    
    # exclusão de colunas sem marcação
    covid_df = covid_df.drop(['ausegia', 'anosmia', 'nausea_vomito',
                        'coriza', 'congestao_nasal', 'calafrio'], axis=1)
    
    #exclusão das instancia cujo fator não foi informado
    covid_df = covid_df[covid_df['fator_nao_informado'] != 'X']
    
    #exclusão da coluna fator_nao_informado
    covid_df = covid_df.drop(['fator_nao_informado'], axis=1)
    
    #exclusão das instancia cujos sintomas não foram informados
    covid_df = covid_df[covid_df['nao_informado'] != 'X']
    
    #exclusão da coluna nao_informado
    covid_df = covid_df.drop(['nao_informado'], axis=1)
    
    #exclusão das instâncias de casos que ainda estão em aberto
    
    #pacientes que estão em isolamento domiciliar
    covid_df = covid_df[covid_df['situacao_atual'] != 'Isolamento Domiciliar']
    
    #pacientes que estão em internação UTI
    covid_df = covid_df[covid_df['situacao_atual'] != 'Internação UTI']
    
    #pacientes que estão em internamento de leite clínico
    covid_df = covid_df[covid_df['situacao_atual'] != 'Internação Leito Clínico']
    
    #Substituição de valores
    
    #Substitue os valores NaN por 0 (zero)
    covid_df = covid_df.fillna(0)
    #Substitue os valores X por 1 
    covid_df = covid_df.replace('X', 1)
    #Substitue valores Masculino e Feminino
    covid_df = covid_df.replace('Masculino', 1)
    covid_df = covid_df.replace('Mascuino', 1)
    covid_df = covid_df.replace('Feminino', 0)
    
    #Substitue valores na coluna situacao_atual
    covid_df = covid_df.replace('Óbito', 1)
    #Substitue todas as outras situacoes em 0 (não morte)
    covid_df = covid_df.replace(to_replace='[a-zA-Z]', value=0, regex=True)
    
    #exclusão das instâncias que não têm nenhum valor
    covid_df = covid_df.drop(covid_df[(covid_df['febre'] == 0) 
                              & (covid_df['tosse'] == 0)
                              & (covid_df['cefaleia'] == 0)
                              & (covid_df['dificuldade_respiratoria'] == 0)
                              & (covid_df['dispineia'] == 0)
                              & (covid_df['mialgia'] == 0)
                              & (covid_df['saturacao_menor_noventa_cinco'] == 0)
                              & (covid_df['adinofagia'] == 0)
                              & (covid_df['diarreia'] == 0)
                              & (covid_df['adinamia'] == 0)
                              & (covid_df['doenca_cardiovascular'] == 0)
                              & (covid_df['diabetes'] == 0)
                              & (covid_df['doenca_respiratoria_cronica'] == 0)
                              & (covid_df['hipertensao'] == 0)
                              & (covid_df['paciente_oncologico'] == 0)
                              & (covid_df['obesidade'] == 0)
                              & (covid_df['doenca_renal_cronica'] == 0)
                              & (covid_df['asma'] == 0)
                              & (covid_df['sem_comorbidade'] == 0)
                              & (covid_df['pneumopatia'] == 0)].index)
    
    # Balanceamento de instâncias
    group = covid_df.groupby(covid_df.situacao_atual)
    covid_df_1 = group.get_group(1)
    covid_df_0 = group.get_group(0)
    
    # Numero total de casos de obitos
    index = covid_df_1.index
    number_of_rows = len(index)
    
    # Seleciona aleatoriamente o numero de linhas
    covid_df_0 = covid_df_0.sample(n = number_of_rows)
    
    # Faz o merge dos dois df
    covid_df = pd.concat([covid_df_1, covid_df_0], ignore_index=True)
    
    # Faz o shuffle do merge dos dfs
    covid_df = covid_df.sample(frac=1).reset_index(drop=True)
    
    #mover coluna situação_atual para o final do dataframe
    situacao = covid_df.pop('situacao_atual')
    
    
    # print(covid_df.info())
    
    #Dividir o conjunto de informações em dados de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(covid_df, situacao, 
                                                        test_size=0.2, 
                                                        random_state=42)
    
    return X_train, X_test, y_train, y_test

