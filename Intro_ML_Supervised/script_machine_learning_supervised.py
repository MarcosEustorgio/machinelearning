# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 15:01:30 2024

@author: Marcos Eustorgio
"""

#Importando pacotes necessarios
import pandas as pd
import numpy as np
import sklearn as skl
import seaborn as sns
import matplotlib.pyplot as plt

#====================Dados Banknote Authentication=============================


#Importando dados
Banknote=pd.read_csv('data_banknote_authentication.txt', sep=',',header=None)

#Renomeando colunas
Banknote.columns=['X1', 'X2', 'X3', 'X4', 'Y']

#Visualização das linhas iniciais
Banknote.head()


#Medidas resumo para as features
Banknote.loc[:,'X1':'X4'].describe().T



#Obtendo proporcao de valores para cada categoria do target
(Banknote['Y'].value_counts()/Banknote.shape[0]).round(2)*100


#Gráficos--------------------------------------


#Gráfico da matriz de correlação
sns.heatmap(Banknote.loc[:,'X1':'X4'].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Matriz de Correlação')
plt.show()



#Investigando correlação em cada grupo de classificação
matriz_correlacao_0=Banknote.loc[Banknote['Y']==0].loc[:,'X1':'X4'].corr()
matriz_correlacao_1=Banknote.loc[Banknote['Y']==1].loc[:,'X1':'X4'].corr()



# Criando a figura e os subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Plotando o primeiro gráfico no primeiro subplot
sns.heatmap(matriz_correlacao_0, annot=True, cmap='coolwarm', fmt='.2f',ax=axs[0])
axs[0].set_title('Matriz de correlação Y=0')

# Plotando o segundo gráfico no segundo subplot
sns.heatmap(matriz_correlacao_1, annot=True, cmap='coolwarm', fmt='.2f',ax=axs[1])
axs[1].set_title('Matriz de correlação Y=1')

#Evitando sobreposição e printando
plt.tight_layout()
plt.show()



# Pairplot usando seaborn

# Alterando tamanho da fonte
sns.set(font_scale=2)
plt.rc('legend',markerscale=3.0)

pairplot = ( sns.pairplot(Banknote, hue='Y', palette='viridis', markers=['o', 'o'],
plot_kws={'s': 20},diag_kind='kde', corner=True, height=3) )
plt.show(pairplot)




#Agora vamos plotar os boxplots para cada valor único de Y 

# Criando um subplot com 1 linha e 4 colunas
fig, axs = plt.subplots(1, 4, figsize=(10, 5))

# Criando um boxplot para cada variável X, separados por valor de Y
for i, col in enumerate(['X1', 'X2', 'X3', 'X4']):
    sns.boxplot(x='Y', y=col, data=Banknote, ax=axs[i])
    plt.tight_layout()
  
# Exibindo o gráfico
plt.show()



# Processo de divisão dos dados -----------------------------------

from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, accuracy_score,matthews_corrcoef,
                            precision_score, recall_score, f1_score,brier_score_loss)

# Seu conjunto de dados
X = Banknote.loc[:,'X1':'X4']  
y = Banknote['Y']


# Divisão em treino e teste com uma semente (seed)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=321)



#Modelo logístico -----------------------------------

from sklearn.linear_model import LogisticRegression

# Modelo de regressão logística
model = LogisticRegression()
reg_log = model.fit(X_train, y_train)

# Coeficientes do modelo
#print("Coeficientes:", reg_log.coef_)
#print("Intercept:", reg_log.intercept_)

# Desempenho preditivo (teste)
y_pred = reg_log.predict(X_test)


# Matriz de confusão
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

# Métricas de comparação
VN, FP, FN, VP = conf_matrix.ravel()
print({'VN': VN, 'FP': FP, 'FN': FN, 'VP':VP})


ACC = accuracy_score(y_test, y_pred)
MCC = matthews_corrcoef(y_test, y_pred)
SEN = recall_score(y_test, y_pred)
VPP = precision_score(y_test, y_pred)
F1 = f1_score(y_test, y_pred)
BS = brier_score_loss(y_test, reg_log.predict_proba(X_test)[:, 1])

# Dicionario com métricas de comparacao para modelo logistico
rlog = {'ACC': ACC, 'MCC': MCC, 'SEN': SEN, 'VPP':VPP, 'F1': F1, 'BS': BS}
rlog = {key: round(value, 2) for key, value in rlog.items()}
print(rlog)




#KNN - Classificação -----------------------------------

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Padronizar as features (opcional, mas recomendado para o KNN)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)



# Criar e treinar o modelo KNN
k = 10  # Número de vizinhos (Escolha inicial arbitraria)
model = KNeighborsClassifier(n_neighbors=k)
knn_c=model.fit(X_train_sc, y_train)

# Fazer previsões no conjunto de teste
y_pred = knn_c.predict(X_test_sc)

# Matriz de confusão
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

# Métricas de comparação
VN, FP, FN, VP = conf_matrix.ravel()
print({'VN': VN, 'FP': FP, 'FN': FN, 'VP':VP})



# Métricas de comparação
ACC = accuracy_score(y_test, y_pred)
MCC = matthews_corrcoef(y_test, y_pred)
SEN = recall_score(y_test, y_pred)
VPP = precision_score(y_test, y_pred)
F1 = f1_score(y_test, y_pred)
BS = brier_score_loss(y_test, knn_c.predict_proba(X_test_sc)[:, 1])


# Dicionario com métricas de comparacao para modelo KNN - Classificacao
knnc = {'ACC': ACC, 'MCC': MCC, 'SEN': SEN, 'VPP':VPP, 'F1': F1, 'BS': BS}
knnc = {key: round(value, 2) for key, value in knnc.items()}
print(knnc)



#Gaussian Naive Bayes (NB) -----------------------------------


from sklearn.naive_bayes import GaussianNB

# Criar e treinar o modelo Gaussian Naive Bayes
model = GaussianNB()
gnbayes=model.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = gnbayes.predict(X_test)


# Matriz de confusão
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

# Métricas de comparação
VN, FP, FN, VP = conf_matrix.ravel()
print({'VN': VN, 'FP': FP, 'FN': FN, 'VP':VP})



# Métricas de comparação
#VN, FP, FN, VP = conf_matrix.ravel()
ACC = accuracy_score(y_test, y_pred)
MCC = matthews_corrcoef(y_test, y_pred)
SEN = recall_score(y_test, y_pred)
VPP = precision_score(y_test, y_pred)
F1 = f1_score(y_test, y_pred)
BS = brier_score_loss(y_test, gnbayes.predict_proba(X_test)[:, 1])


# Dicionario com métricas de comparacao para modelo GNB - Classificacao
gnb = {'ACC': ACC, 'MCC': MCC, 'SEN': SEN, 'VPP':VPP, 'F1': F1, 'BS': BS}
gnb = {key: round(value, 2) for key, value in gnb.items()}
print(gnb)



# Árvore de Decisão ----------------------------------------------------------

from sklearn.tree import DecisionTreeClassifier, plot_tree

# Criar e treinar o modelo de árvore de decisão
model = DecisionTreeClassifier(criterion='entropy')
DTClass=model.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = DTClass.predict(X_test)


# Matriz de confusão
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)


# Métricas de comparação
#VN, FP, FN, VP = conf_matrix.ravel()
ACC = accuracy_score(y_test, y_pred)
MCC = matthews_corrcoef(y_test, y_pred)
SEN = recall_score(y_test, y_pred)
VPP = precision_score(y_test, y_pred)
F1 = f1_score(y_test, y_pred)
BS = brier_score_loss(y_test, DTClass.predict_proba(X_test)[:, 1])


# Plotar a árvore de decisão
plt.figure(figsize=(20, 20))  # Definir o tamanho da figura
plot_tree(DTClass, filled=True, feature_names=['X1', 'X2', 'X3', 'X4'],
class_names=['0', '1'], rounded=True, fontsize=15)
plt.show()


# Dicionario com métricas de comparacao para modelo Arvore de Decisao - Classificacao
dtc = {'ACC': ACC, 'MCC': MCC, 'SEN': SEN, 'VPP':VPP, 'F1': F1, 'BS': BS}
dtc = {key: round(value, 2) for key, value in dtc.items()}
print(dtc)


#Comparativo entre modelos de classificação -----------------------------------------------------

# Criar um DataFrame a partir dos dicionários e adicionar uma coluna com o nome do dicionário
comp_model_class = pd.DataFrame([rlog, knnc, gnb, dtc])
comp_model_class['models'] = ['Reg. Logistic', 'KNN Class', 'Gaussian NB', 'DecisionTree Class']

# Exibir o DataFrame
print(comp_model_class)


# Ordenar o DataFrame pelos valores das colunas 'ACC' a 'F1' em ordem decrescente e pela
#coluna 'BS' em ordem crescente:
sorted_df = comp_model_class.sort_values(by=['ACC', 'MCC', 'SEN', 'VPP', 'F1', 'BS'],
ascending=[False, False, False, False, False, True])

# Imprimir o DataFrame ordenado
print(sorted_df)


# Ajuste do modelo de regressão logística a todo conjunto de dados ---------------------------------

# Modelo de regressão logística
model = LogisticRegression()
reg_log = model.fit(X, y)

# Coeficientes do modelo
coeficientes = reg_log.coef_[0]
intercepto = reg_log.intercept_
odds_ratio = np.exp(coeficientes)

print(intercepto)

# Criar um dicionário com os dados e rótulos
df_dict = {'Coeficientes': coeficientes, 'Odds Ratio': odds_ratio}

# Criar um DataFrame a partir do dicionário
tabela_coeficientes = pd.DataFrame(df_dict, index=X.columns.tolist())


# Imprimindo a tabela de coeficientes
print(tabela_coeficientes)






#=============================================== DADOS BOSTON ==============================================


#Importando dados Boston
Boston=pd.read_excel('Boston_Housing.xlsx')

# Definir a configuração para mostrar todas as colunas
pd.set_option('display.max_columns', None)

#Visualização da estrutura dos dados
print(Boston.head())


# Análise descritiva e exploratória -------------------------------------


#Medidas resumo para variaveis dos dados Boston
Boston.describe().T


#Proporção de observações nas categorias da feature chas
Boston['chas'].value_counts()




#Gráfico da matriz de correlação
sns.set(font_scale=1.0)
pairplot=(sns.heatmap(Boston.drop(columns=['chas']).corr(), annot=True,
cmap='coolwarm', fmt='.2f',annot_kws={"size": 8}))
plt.title('Matriz de Correlação')
plt.show(pairplot)


#Gráfico de dispersão entre variaveis da base de dados
# A variavel 'chas' é categorica e por isso nao sera considerada neste grafico

# Setting the font size by 3 
sns.set(font_scale=3)

pairplot=(sns.pairplot(Boston.drop(columns=['chas']), markers=['o', 'o'],
plot_kws={'s': 30},diag_kind='kde', corner=True, height=3,palette='black'))
plt.show(pairplot)



# Plotar os boxplots
plt.figure(figsize=(8, 8))
sns.set(font_scale=1.5)
sns.boxplot(data=Boston.drop(columns=['chas']), orient='h')
plt.show()


boston_df=Boston.drop(columns=['chas','medv'])

# Criando uma figura e os subplots
fig, axs = plt.subplots(3, 4, figsize=(10, 10))

# Iterando sobre as variáveis e criando um boxplot para cada uma delas
for i, colu in enumerate(boston_df.columns):
    row = i // 4
    col = i % 4
    sns.boxplot(y=boston_df[colu], ax=axs[row, col])

# Ajustando o layout e mostrando os gráficos
plt.tight_layout()
plt.show()



# Processo de divisão dos dados -------------------------------------------


from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# Seu conjunto de dados
X = Boston.drop(columns=['medv'])  
y = Boston['medv']


# Divisão em treino e teste com uma semente (seed)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=321)



# Padronizar as features ( recomendado para estes dados)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)


# Regressão linear múltipla -----------------------------------------------------


from sklearn.linear_model import LinearRegression

# Atribuindo modelo de regressão linear múltipla a um objeto
lm = LinearRegression()

# Treinando o modelo com dados de treino
lm.fit(X_train, y_train)

# Calcular previsões utilizando dados de teste
y_pred = lm.predict(X_test)

# Calcular métricas de ajuste
r2 = r2_score(y_test, y_pred)
adj_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))


# Dicionario contendo métricas de comparacao para modelo de Reg. Linear multipla
reg_lin = {'R2': r2, 'R2a': adj_r2, 'MAE': mae, 'MAPE':mape, 'RMSE': rmse}
reg_lin = {key: round(value, 2) for key, value in reg_lin.items()}
print(reg_lin)



# KNN - Regressão ----------------------------------------------------------------


from sklearn.neighbors import KNeighborsRegressor

# Atribuindo modelo de KNN para Regressão (k=10 arbitrário) a um objeto
knn_reg = KNeighborsRegressor(n_neighbors=10)

# Treinando o modelo com dados de treino
knn_reg.fit(X_train, y_train)

# Calcular previsões utilizando dados de teste
y_pred = knn_reg.predict(X_test)

# Calcular métricas de ajuste
r2 = r2_score(y_test, y_pred)
adj_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))


# Dicionario contendo métricas de comparacao para modelo de KNN Regressão
KNN_Reg = {'R2': r2, 'R2a': adj_r2, 'MAE': mae, 'MAPE':mape, 'RMSE': rmse}
KNN_Reg = {key: round(value, 2) for key, value in KNN_Reg.items()}
print(KNN_Reg)


# Árvore de regressão -----------------------------------------------------------

from sklearn.tree import DecisionTreeRegressor

# Atribuindo modelo de Árvore de Regressão a um objeto
arvre_reg = DecisionTreeRegressor()

# Treinando o modelo com dados de treino
arvre_reg.fit(X_train, y_train)

# Calcular previsões utilizando dados de teste
y_pred = arvre_reg.predict(X_test)


# Calcular métricas de ajuste
r2 = r2_score(y_test, y_pred)
adj_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Dicionario contendo métricas de comparacao para modelo de Árvore de Regressão
Arvre_Reg = {'R2': r2, 'R2a': adj_r2, 'MAE': mae, 'MAPE':mape, 'RMSE': rmse}
Arvre_Reg = {key: round(value, 2) for key, value in Arvre_Reg.items()}
print(Arvre_Reg)


# Comparativo entre modelos ------------------------------------------------------


# Criar um DataFrame a partir dos dicionários e adicionar uma coluna com o nome do dicionário
comp_model_reg = pd.DataFrame([reg_lin, KNN_Reg, Arvre_Reg])
comp_model_reg['models'] = ['Reg. Linear', 'KNN Regressão', 'Árvore de Regressão']

# Exibir o DataFrame
print(comp_model_reg)


# Ordenar o DataFrame pelos valores das colunas 'ACC' a 'F1' em ordem decrescente e pela
#coluna 'BS' em ordem crescente:
sorted_df = comp_model_reg.sort_values(by=['R2', 'R2a', 'MAE', 'MAPE', 'RMSE'],
ascending=[False, False, True, True, True])

# Imprimir o DataFrame ordenado
print(sorted_df)


#Ajustando modelo de regressão linear aos dados completos
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Padronizando as features
scaler = StandardScaler()
X_sc = scaler.fit_transform(X)

# Atribuindo modelo de regressão linear múltipla a um objeto
lm_all = LinearRegression()

# Treinando o modelo com dados de treino
lm_all.fit(X_sc, y)

# Obtendo os coeficientes do modelo
coeficientes = lm_all.coef_
intercepto = lm_all.intercept_


# Criando a tabela de coeficientes
tabela_coeficientes = pd.DataFrame(coeficientes, columns=['Coeficiente'], index=X.columns.tolist())
tabela_coeficientes.loc['Intercepto'] = intercepto


# Imprimindo a tabela de coeficientes
print(tabela_coeficientes)
