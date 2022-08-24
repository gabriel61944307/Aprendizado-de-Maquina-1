import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

dados = load_breast_cancer()
X = dados.data
Y = dados.target

dados_treino, dados_teste, rotulos_treino, rotulos_teste = train_test_split(X, Y,
                                          test_size = 0.2,
                                          random_state=42,
                                          stratify=Y)

#Arvore de decisão
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()
dtc.fit(dados_treino, rotulos_treino)

rotulos_preditos = dtc.predict(dados_teste)
accuracy_score(rotulos_teste, rotulos_preditos)

tr_acc = []
tr_std = []

for this_md in range(2,30):
  dtc = DecisionTreeClassifier(max_depth=this_md)
  dtc.fit(dados_treino, rotulos_treino)
  scores = cross_val_score(dtc, dados_treino, rotulos_treino, cv=10)
  tr_acc.append(scores.mean())
  tr_std.append(np.std(scores))
  
plt.errorbar(x=range(2,30), y=tr_acc, yerr=tr_std)
  
from sklearn.model_selection import GridSearchCV
param_grid = {'criterion': ['entropy', 'gini'],
              'max_depth': range(2,30,2),
              'min_samples_leaf': range(2,10,2),
              'min_impurity_decrease': np.linspace(0,0.5,10)}

dtc = DecisionTreeClassifier()
gs = GridSearchCV(dtc, param_grid=param_grid)
gs.fit(dados_treino, rotulos_treino)

gs.best_estimator_

dtc = DecisionTreeClassifier(criterion='entropy', max_depth=20, min_samples_leaf=8)
dtc.fit(dados_treino, rotulos_treino)
rotulos_preditos = dtc.predict(dados_teste)
accuracy_score(rotulos_teste, rotulos_preditos)
