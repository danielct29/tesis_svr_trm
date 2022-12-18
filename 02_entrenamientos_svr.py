"""
Entremientos RSV
"""

import time
inicio = time.time()


import pandas as pd
import datetime as dt
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt


#%% Importaccion de datos arreglados
ruta = '/Datos/Datos_arreglados.xlsx'
df = pd.read_excel(ruta)
df.set_index('FECHA', inplace=True)
df.sort_index(ascending=True, inplace=True)
df.dropna(axis=0, inplace=True)
df = df.loc[dt.date(2012, 1, 2):, :].copy()


#%% Segmentacion de datos

#### Idenfiticacion de Variables X y Y
y_cols = ['RETORNO_T_0']

x_cols_1 = [
    'RETORNO_T_1', 'RETORNO_T_2', 'RETORNO_T_3', 'RETORNO_T_4', 'RETORNO_T_5'
    ]

x_cols_2 = [
    'KAMA T_1', 'RSI T_1', 'STO T_1', 'WR T_1', 'ADX T_1', 'CCI T_1',
    'EMA T_1', 'MACD T_1', 'SMA T_1', 'ATR T_1', 'CMF T_1', 'OBV T_1'
    ]

x_cols_3 = x_cols_1 + x_cols_2

#### Segmentacion en set de entrenamiento y testeo
x_train, x_test, y_train, y_test = train_test_split(df[x_cols_3], df[y_cols],
                                                    train_size=0.8,
                                                    shuffle=False)


#%% Definicion de Modelos

from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import FastICA
from sklearn.svm import SVR

#### Histogramas
fig, axes = plt.subplots(1, 2, figsize=(20,6), sharex=False)
y_train.hist()
y_test.hist()


#### Pipelines
pipe_1 = Pipeline([
    ('scaler', StandardScaler()),
    ('svr', SVR(cache_size=1000, shrinking=True))])

pipe_2 = Pipeline([
    ('scaler', StandardScaler()),
    ('feat', PCA(whiten=False, tol=1e-3)),
    ('svr', SVR(cache_size=1000, shrinking=True))])

pipe_3 = Pipeline([
    ('scaler', StandardScaler()),
    ('feat',KernelPCA(tol=1e-3, eigen_solver='arpack')),
    ('svr', SVR(cache_size=1000, shrinking=True))])

pipe_4 = Pipeline([
    ('scaler', StandardScaler()),
    ('feat', FastICA(whiten=True, tol=1e-3, max_iter=1000, algorithm = 'deflation')),
    ('svr', SVR(cache_size=1000, shrinking=True))])

#### Parametros de entrenamiento
n_jobs = 10
scoring = {'Costo': 'neg_mean_squared_error', 'Precision': 'r2'}
refit = 'Costo'
verb = 1
kf = KFold(n_splits=3, shuffle=False)

#### Parametros de modelacion
# Parametro de regularizacion C
c = [1, 10, 50, 100]
# Margen de error epsilon
epsilon = [0.01, 0.05, 0.1, 0.5]
# Funciones Kernel de SVR
kernel = ['rbf', 'sigmoid', 'poly']
kernel_kpca = ['rbf', 'sigmoid']
# Gamma de la funcion
gamma = ['scale']

# N Componentes
n_comp_1 = list(range(1, len(x_cols_1)+1))
n_comp_2 = list(range(1, len(x_cols_2)+1))
n_comp_3 = list(range(1, len(x_cols_3)+1))
# n_comp_4 = list(range(1, len(x_cols_4)+1))


Param_svr = {
    'svr__C': c,
    'svr__epsilon': epsilon,
    'svr__kernel': kernel,
    'svr__gamma':gamma,
    }

Param_pca_svr_1 = {
    'feat__n_components': n_comp_1, 'svr__C': c, 'svr__epsilon': epsilon,
    'svr__kernel': kernel, 'svr__gamma':gamma
    }

Param_pca_svr_2 = {
    'feat__n_components': n_comp_2, 'svr__C': c, 'svr__epsilon': epsilon,
    'svr__kernel': kernel, 'svr__gamma':gamma
    }

Param_pca_svr_3 = {
    'feat__n_components': n_comp_3, 'svr__C': c, 'svr__epsilon': epsilon,
    'svr__kernel': kernel, 'svr__gamma':gamma
    }

Param_kpca_svr_1 = {
    'feat__kernel':kernel_kpca, 'feat__n_components': n_comp_1,
    'svr__C': c, 'svr__epsilon': epsilon,
    'svr__kernel': kernel, 'svr__gamma':gamma
    }

Param_kpca_svr_2 = {
    'feat__kernel':kernel_kpca, 'feat__n_components': n_comp_2,
    'svr__C': c, 'svr__epsilon': epsilon,
    'svr__kernel': kernel, 'svr__gamma':gamma
    }

Param_kpca_svr_3 = {
    'feat__kernel':kernel_kpca, 'feat__n_components': n_comp_3,
    'svr__C': c, 'svr__epsilon': epsilon,
    'svr__kernel': kernel, 'svr__gamma':gamma
    }

Param_ica_svr_1 = {
    'feat__n_components': n_comp_1, 'svr__C': c, 'svr__epsilon': epsilon,
    'svr__kernel': kernel, 'svr__gamma':gamma
    }

Param_ica_svr_2 = {
    'feat__n_components': n_comp_2, 'svr__C': c, 'svr__epsilon': epsilon,
    'svr__kernel': kernel, 'svr__gamma':gamma
    }

Param_ica_svr_3 = {
    'feat__n_components': n_comp_3, 'svr__C': c, 'svr__epsilon': epsilon,
    'svr__kernel': kernel, 'svr__gamma':gamma
    }

#### SVR
model_1_1 = GridSearchCV(
    pipe_1, Param_svr, cv=kf, scoring = scoring, refit = refit,
    return_train_score = False, verbose = verb, n_jobs = n_jobs)

model_1_2 = GridSearchCV(
    pipe_1, Param_svr, cv=kf, scoring = scoring, refit = refit,
    return_train_score = False, verbose = verb, n_jobs = n_jobs)

model_1_3 = GridSearchCV(
    pipe_1, Param_svr, cv=kf, scoring = scoring, refit = refit,
    return_train_score = False, verbose = verb, n_jobs = n_jobs)


#### PCA + SVR
model_2_1 = GridSearchCV(
    pipe_2, Param_pca_svr_1, cv=kf, scoring = scoring, refit = refit,
    return_train_score = False, verbose = verb, n_jobs = n_jobs)

model_2_2 = GridSearchCV(
    pipe_2, Param_pca_svr_2, cv=kf, scoring = scoring, refit = refit,
    return_train_score = False, verbose = verb, n_jobs = n_jobs)

model_2_3 = GridSearchCV(
    pipe_2, Param_pca_svr_3, cv=kf, scoring = scoring, refit = refit,
    return_train_score = False, verbose = verb, n_jobs = n_jobs)


#### KPCA + SVR
model_3_1 = GridSearchCV(
    pipe_3, Param_kpca_svr_1, cv=kf, scoring = scoring, refit = refit,
    return_train_score = False, verbose = verb, n_jobs = n_jobs)

model_3_2 = GridSearchCV(
    pipe_3, Param_kpca_svr_2, cv=kf, scoring = scoring, refit = refit,
    return_train_score = False, verbose = verb, n_jobs = n_jobs)

model_3_3 = GridSearchCV(
    pipe_3, Param_kpca_svr_3, cv=kf, scoring = scoring, refit = refit,
    return_train_score = False, verbose = verb, n_jobs = n_jobs)


#### ICA + SVR
model_4_1 = GridSearchCV(
    pipe_4, Param_ica_svr_1, cv=kf, scoring = scoring, refit = refit,
    return_train_score = False, verbose = verb, n_jobs = n_jobs)


model_4_2 = GridSearchCV(
    pipe_4, Param_ica_svr_2, cv=kf, scoring = scoring, refit = refit,
    return_train_score = False, verbose = verb, n_jobs = n_jobs)


model_4_3 = GridSearchCV(
    pipe_4, Param_ica_svr_3, cv=kf, scoring = scoring, refit = refit,
    return_train_score = False, verbose = verb, n_jobs = n_jobs)



#%% Entramiento

import pickle

r_save = '/Modelos/'
#### SVR
model_1_1.fit(x_train[x_cols_1], y_train[y_cols[0]].values)
pickle.dump(model_1_1, open(r_save + 'model_1_1.pkl', 'wb'))

model_1_2.fit(x_train[x_cols_2], y_train[y_cols[0]].values)
pickle.dump(model_1_2, open(r_save + 'model_1_2.pkl', 'wb'))

model_1_3.fit(x_train[x_cols_3], y_train[y_cols[0]].values)
pickle.dump(model_1_3, open(r_save + 'model_1_3.pkl', 'wb'))


#### PCA + SVR 
model_2_1.fit(x_train[x_cols_1], y_train[y_cols[0]].values)
pickle.dump(model_2_1, open(r_save + 'model_2_1.pkl', 'wb'))

model_2_2.fit(x_train[x_cols_2], y_train[y_cols[0]].values)
pickle.dump(model_2_2, open(r_save + 'model_2_2.pkl', 'wb'))

model_2_3.fit(x_train[x_cols_3], y_train[y_cols[0]].values)
pickle.dump(model_2_3, open(r_save + 'model_2_3.pkl', 'wb'))


#### KPCA + SVR
model_3_1.fit(x_train[x_cols_1], y_train[y_cols[0]].values)
pickle.dump(model_3_1, open(r_save + 'model_3_1.pkl', 'wb'))

model_3_2.fit(x_train[x_cols_2], y_train[y_cols[0]].values)
pickle.dump(model_3_2, open(r_save + 'model_3_2.pkl', 'wb'))

model_3_3.fit(x_train[x_cols_3], y_train[y_cols[0]].values)
pickle.dump(model_3_3, open(r_save + 'model_3_3.pkl', 'wb'))


#### ICA + SVR
model_4_1.fit(x_train[x_cols_1], y_train[y_cols[0]].values)
pickle.dump(model_4_1, open(r_save + 'model_4_1.pkl', 'wb'))

model_4_2.fit(x_train[x_cols_2], y_train[y_cols[0]].values)
pickle.dump(model_4_2, open(r_save + 'model_4_2.pkl', 'wb'))

model_4_3.fit(x_train[x_cols_3], y_train[y_cols[0]].values)
pickle.dump(model_4_3, open(r_save + 'model_4_3.pkl', 'wb'))





fin = time.time()
print('tiempo ejecucion: ', fin-inicio)



