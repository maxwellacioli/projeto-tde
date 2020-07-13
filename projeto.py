# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 08:41:30 2020

@author: Acioli
"""


import pandas as pd
import numpy as np

covid = pd.read_csv("covid19-al-sintomas.csv")

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
# covid = covid[covid['situacao_atual'] != 'Isolamento Domiciliar']

#pacientes que estão em internação UTI
covid = covid[covid['situacao_atual'] != 'Internação UTI']

#pacientes que estão em internamento de leite clínico
covid = covid[covid['situacao_atual'] != 'Internação Leito Clínico']


print(covid.info())