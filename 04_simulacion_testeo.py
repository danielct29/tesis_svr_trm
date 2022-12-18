"""
Simulacion Testeo
"""
import pandas as pd
import pickle
import time

r_save = '/Datos/'

modelos = {}
with (open(r_save + 'model_1_1.pkl', "rb")) as openfile:
    modelos.update({'model_1_1': pickle.load(openfile)})

with (open(r_save + 'model_1_2.pkl', "rb")) as openfile:
    modelos.update({'model_1_2': pickle.load(openfile)})

with (open(r_save + 'model_1_3.pkl', "rb")) as openfile:
    modelos.update({'model_1_3': pickle.load(openfile)})


with (open(r_save + 'model_2_1.pkl', "rb")) as openfile:
    modelos.update({'model_2_1': pickle.load(openfile)})

with (open(r_save + 'model_2_2.pkl', "rb")) as openfile:
    modelos.update({'model_2_2': pickle.load(openfile)})

with (open(r_save + 'model_2_3.pkl', "rb")) as openfile:
    modelos.update({'model_2_3': pickle.load(openfile)})


with (open(r_save + 'model_3_1.pkl', "rb")) as openfile:
    modelos.update({'model_3_1': pickle.load(openfile)})

with (open(r_save + 'model_3_2.pkl', "rb")) as openfile:
    modelos.update({'model_3_2': pickle.load(openfile)})

with (open(r_save + 'model_3_3.pkl', "rb")) as openfile:
    modelos.update({'model_3_3': pickle.load(openfile)})


with (open(r_save + 'model_4_1.pkl', "rb")) as openfile:
    modelos.update({'model_4_1': pickle.load(openfile)})

with (open(r_save + 'model_4_2.pkl', "rb")) as openfile:
    modelos.update({'model_4_2': pickle.load(openfile)})

with (open(r_save + 'model_4_3.pkl', "rb")) as openfile:
    modelos.update({'model_4_3': pickle.load(openfile)})


tipo_model = {
    '1': 'SVR',
    '2': 'PCA+SVR',
    '3': 'KPCA+SVR',
    '4': 'ICA+SVR',
    }

scores = {}
for k, v in modelos.items():
    tipo_model.get(k[6])
    scores.update({k: {'Score': v.best_score_,
                       'Model': tipo_model.get(k[6]),
                       'Set Datos': k[-1]}})
    # print(k, v)

    df_results = pd.DataFrame.from_dict(v.cv_results_)
    df_results.sort_values(by='rank_test_Costo', ascending=True, inplace=True)
    df_results.to_excel(r_save + 'Resultados_' + k + '.xlsx')


df_scores = pd.DataFrame.from_dict(scores, orient='index')
df_scores = df_scores.pivot(index='Model', columns='Set Datos', values='Score')
df_scores = df_scores.mul(-1)
df_scores.to_excel(r_save+'scores.xlsx')


#%% Mejores Modelos

model_1_1 = modelos.get('model_1_1').best_estimator_
model_1_2 = modelos.get('model_1_2').best_estimator_
model_1_3 = modelos.get('model_1_3').best_estimator_

model_2_1 = modelos.get('model_2_1').best_estimator_
model_2_2 = modelos.get('model_2_2').best_estimator_
model_2_3 = modelos.get('model_2_3').best_estimator_

model_3_1 = modelos.get('model_3_1').best_estimator_
model_3_2 = modelos.get('model_3_2').best_estimator_
model_3_3 = modelos.get('model_3_3').best_estimator_

model_4_1 = modelos.get('model_4_1').best_estimator_
model_4_2 = modelos.get('model_4_2').best_estimator_
model_4_3 = modelos.get('model_4_3').best_estimator_


#%% Datos de testeo
import datetime as dt

#### Importaccion de datos arreglados
ruta = '/Users/danielc./Documents/Code/Tesis_svr/Datos/Datos_arreglados.xlsx'
df = pd.read_excel(ruta)
df.set_index('FECHA', inplace=True)
df.sort_index(ascending=True, inplace=True)
df.dropna(axis=0, inplace=True)
df = df.loc[dt.date(2012, 1, 2):, :].copy()


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


#%% Simulacion
print('Iniciando simulacion ... \n')
pronosticos = {}
t_inicial = 1931


#### Iteracion
t_final = len(df.index)-1

for i in range(t_inicial, t_final):
    
    inicio = time.time()
    print('entrenando hasta:', df.index[i])
    print('pronostico para:', df.index[i+1])
    
    df_new = df.loc[df.index[0]:df.index[i], :]
    df_pron = df.loc[df.index[i+1], :].to_frame().transpose()
    
    #### Entrenamiento
    print('entramiento ...')
    model_1_1.fit(df_new[x_cols_1], df_new[y_cols[0]].values)
    model_1_2.fit(df_new[x_cols_2], df_new[y_cols[0]].values)
    model_1_3.fit(df_new[x_cols_3], df_new[y_cols[0]].values)
    
    model_2_1.fit(df_new[x_cols_1], df_new[y_cols[0]].values)
    model_2_2.fit(df_new[x_cols_2], df_new[y_cols[0]].values)
    model_2_3.fit(df_new[x_cols_3], df_new[y_cols[0]].values)
    
    model_3_1.fit(df_new[x_cols_1], df_new[y_cols[0]].values)
    model_3_2.fit(df_new[x_cols_2], df_new[y_cols[0]].values)
    model_3_3.fit(df_new[x_cols_3], df_new[y_cols[0]].values)
    
    model_4_1.fit(df_new[x_cols_1], df_new[y_cols[0]].values)
    model_4_2.fit(df_new[x_cols_2], df_new[y_cols[0]].values)
    model_4_3.fit(df_new[x_cols_3], df_new[y_cols[0]].values)
    
    #### Pronostico
    print('pronostico ...')
    y_pred_1_1 = model_1_1.predict(df_pron[x_cols_1])
    y_pred_1_2 = model_1_2.predict(df_pron[x_cols_2])
    y_pred_1_3 = model_1_3.predict(df_pron[x_cols_3])
    
    y_pred_2_1 = model_2_1.predict(df_pron[x_cols_1])
    y_pred_2_2 = model_2_2.predict(df_pron[x_cols_2])
    y_pred_2_3 = model_2_3.predict(df_pron[x_cols_3])
    
    y_pred_3_1 = model_3_1.predict(df_pron[x_cols_1])
    y_pred_3_2 = model_3_2.predict(df_pron[x_cols_2])
    y_pred_3_3 = model_3_3.predict(df_pron[x_cols_3])
    
    y_pred_4_1 = model_4_1.predict(df_pron[x_cols_1])
    y_pred_4_2 = model_4_2.predict(df_pron[x_cols_2])
    y_pred_4_3 = model_4_3.predict(df_pron[x_cols_3])
    
    
    pronosticos.update(
        {i:
         {'y_true': df_new[y_cols[0]].values[-1],
          'y_pred_1_1': y_pred_1_1[-1],
          'y_pred_1_2': y_pred_1_2[-1],
          'y_pred_1_3': y_pred_1_3[-1],
          'y_pred_2_1': y_pred_2_1[-1],
          'y_pred_2_2': y_pred_2_2[-1],
          'y_pred_2_3': y_pred_2_3[-1],
          'y_pred_3_1': y_pred_3_1[-1],
          'y_pred_3_2': y_pred_3_2[-1],
          'y_pred_3_3': y_pred_3_3[-1],
          'y_pred_4_1': y_pred_4_1[-1],
          'y_pred_4_2': y_pred_4_2[-1],
          'y_pred_4_3': y_pred_4_3[-1]
          }
         }
            )
    
    print(df_new.shape[0], 'de', 2416, '- faltan:', 2416-df_new.shape[0])
    fin = time.time() - inicio
    print('tiempo ejecucion:', fin, '\n')


df_pronosticos = pd.DataFrame.from_dict(pronosticos).transpose()

