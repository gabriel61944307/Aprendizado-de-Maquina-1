import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

param_grid = {'weights': ['uniform', 'distance'],
              'n_neighbors': range(1,15,2),
              'metric': ['euclidean', 'manhattan']}

knc = KNeighborsClassifier()
gs = GridSearchCV(knc, param_grid=param_grid)
gs.fit(dados_treino, rotulos_treino)

gs.best_estimator_

knc = KNeighborsClassifier(metric='manhattan', n_neighbors=11, weights='distance')
knc.fit(dados_treino, rotulos_treino)
rotulos_preditos = knc.predict(dados_teste)
accuracy_score(rotulos_teste, rotulos_preditos)

#normalização
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(dados_treino)
dados_treino_normalizado = scaler.transform(dados_treino)
print(dados_treino_normalizado)

param_grid = {'weights': ['uniform', 'distance'],
              'n_neighbors': range(1,15,2),
              'metric': ['euclidean', 'manhattan']}

knc2 = KNeighborsClassifier()
gs = GridSearchCV(knc2, param_grid=param_grid)
gs.fit(dados_treino_normalizado, rotulos_treino)
gs.best_estimator_

knc2 = KNeighborsClassifier(metric='euclidean', n_neighbors=3)
knc2.fit(dados_treino_normalizado, rotulos_treino)
rotulos_preditos = knc2.predict(scaler.transform(dados_teste))
accuracy_score(rotulos_teste, rotulos_preditos)
