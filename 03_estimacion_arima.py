"""
Estimacion ARIMA
"""

import time
inicio = time.time()

import pandas as pd
import datetime as dt

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, zivot_andrews
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt


#%% Importaccion de datos arreglados
ruta = '/Datos/Datos_arreglados.xlsx'
df = pd.read_excel(ruta)
df.set_index('FECHA', inplace=True)
df.sort_index(ascending=True, inplace=True)
df.dropna(axis=0, inplace=True)
df = df.loc[dt.date(2012, 1, 2):, :].copy()


#%% Segmentacion de datos

#### Idenfiticacion de Variable Dependeinte
y_cols = ['RETORNO_T_0']
y = df[y_cols]

#### 80% y 20%
n = len(df.index)
y_train = y[:int(n*0.8)]
y_test = y[int(n*0.8):]


#%% Identificación

#%%% Estacionariedad (orden de diferencias)

fig, axes = plt.subplots(1, 2, figsize=(15,5), sharex=False)
axes[0].plot(y_train.index, y_train['RETORNO_T_0'],'k-')
axes[0].set_title('Serie Original')
axes[1].plot(y_train.index, y_train['RETORNO_T_0'].diff(1),'k-')
axes[1].set_title('1ra Diferencia')
plt.show()


estacionariedad = {}

#### Dickey Fuller Aumentada
adf_test = adfuller(y_train['RETORNO_T_0'].dropna())
print('ADF Statistic: %f' % adf_test[0])
print('p-value: %f' % adf_test[1])
print('---')
estacionariedad.update(
    {'Dickey Fuller Aumentada':{
        'Estadístico': adf_test[0],
        'P valor': adf_test[1]}
     })

#### Zivot Andrews
za_test = zivot_andrews(y_train['RETORNO_T_0'].dropna())
print('Zivot Andrews Statistic: %f' % za_test[0])
print('p-value: %f' % za_test[1])
print('---')
estacionariedad.update(
    {'Zivot Andrews':{
        'Estadístico': za_test[0],
        'P valor': za_test[1]}
     })

df_estacionariedad = pd.DataFrame.from_dict(estacionariedad, orient='index')
df_estacionariedad.index.name = 'Test'
df_estacionariedad.reset_index(inplace=True)

# Exportar a Latex
# r  = 'Tabla_py_Estacionariedad.tex'
# df_estacionariedad.to_latex(r, index=False, float_format="{:0.5f}".format,
#                             column_format='ccc')


orden_i = 0

#%%% Orden

lags = 10
alpha = 0.01

index_lags = ['Rezago ' + str(i) for i in range(lags+1)]

def _valida_intervalo(minimo, maximo):
    if (minimo < 0) & (0 < maximo):
        y = 'No significativo'
    else:
        y = 'Significativo'
    return y


#### Autocorrelacion Parcial
pacf_result, pacf_intervals = pacf(y_train.dropna(), nlags=lags, alpha=alpha)
print(pacf_result)
plot_pacf(y_train.dropna(), title='Autorcorrelación Parcial', lags=lags,
          alpha = alpha)
plot_pacf(y_train.dropna(), title='', lags=lags, alpha = alpha)

# Resumen PACF
df_pacf = pd.DataFrame(pacf_result, index=index_lags, columns=['PACF'])
df_pacf_intervals = pd.DataFrame(pacf_intervals,
                                 index=index_lags,
                                 columns=['Min. intervalo',
                                          'Max. intervalo'])
df_pacf_summary = pd.concat([df_pacf, df_pacf_intervals], axis=1)
df_pacf_summary['Significancia'] = df_pacf_summary.apply(
    lambda x: _valida_intervalo(x['Min. intervalo'],
                                x['Max. intervalo']),
    axis = 1)
df_pacf_summary.index.name = 'Rezago'
df_pacf_summary.reset_index(inplace=True)

# Exportar a Latex
# r  = 'Tabla_py_PACF.tex'
# df_pacf_summary.to_latex(r, index=False, float_format="{:0.4f}".format,
#                          column_format='ccccc')



#### Autocorrelacion
acf_result, acf_intervals = acf(y_train.dropna(), nlags=lags, alpha=alpha)
print(acf_result)
plot_acf(y_train.dropna(), title='Autorcorrelación', lags=lags,
         alpha = alpha)
plot_acf(y_train.dropna(), title='', lags=lags, alpha = alpha)

# Resumen PACF
df_acf = pd.DataFrame(acf_result, index=index_lags, columns=['ACF'])
df_acf_intervals = pd.DataFrame(acf_intervals,
                                index=index_lags,
                                columns=['Min. intervalo',
                                         'Max. intervalo'])
df_acf_summary = pd.concat([df_acf, df_acf_intervals], axis=1)
df_acf_summary['Significancia'] = df_acf_summary.apply(
    lambda x: _valida_intervalo(x['Min. intervalo'],
                                x['Max. intervalo']),
    axis = 1)
df_acf_summary.index.name = 'Rezago'
df_acf_summary.reset_index(inplace=True)

# Exportar a Latex
# r  = 'Tabla_py_ACF.tex'
# df_acf_summary.to_latex(r, index=False, float_format="{:0.4f}".format,
#                         column_format='ccccc')



#### Tablas resumida
df_acf = pd.DataFrame(acf_result, index=index_lags, columns=['ACF'])
df_acf_intervals = pd.DataFrame(acf_intervals,
                                index=index_lags,
                                columns=['ACF min. intervalo',
                                         'ACF max. intervalo'])

df_cf = pd.concat([df_pacf, df_pacf_intervals, df_acf, df_acf_intervals],
                  axis=1)

orden_ar = 2
orden_ma = 1


#%% Estimacion

print('Estimando ...')
resultados = {}

for ar in range(orden_ar+1):
    for ma in range(orden_ma+1):
        
        orden = (ar, orden_i, ma)
        orden_str = str(ar) + ', ' + str(orden_i) + ', ' + str(ma)
        print(orden_str)
        
        model = ARIMA(y_train.values, order=orden)
        model_fit = model.fit()

        aic = model_fit.aic
        bic = model_fit.bic

        
        resultados.update({orden_str:
                           {'AIC': aic}})

df_resultados = pd.DataFrame.from_dict(resultados, orient='index')
df_resultados.sort_values(by=['AIC'], axis=0, ascending=True, inplace=True)
df_resultados.index.name = 'Orden ARIMA'
df_resultados.reset_index(inplace=True)


# Exportar a Latex
# r  = 'Tabla_py_resultados_AIC_ARIMA.tex'
# df_resultados.to_latex(r, index=False, float_format="{:0.2f}".format)


best_order_str = df_resultados['Orden ARIMA'][0]
best_ar, best_d, best_ma = best_order_str.split(',') 
best_ar = int(best_ar)
best_d = int(best_d)
best_ma = int(best_ma)



#%%% Diagnostico
cols = {
        'Unnamed: 0': 'Variable',
        'coef': 'Coeficiente',
        'std err': 'Desviación',
        '[0.025': 'Inf. (95%)',
        '0.975]': 'Sup. (95%)'
        }

#### Mejor modelo con contante
model = ARIMA(y_train.values, order=(best_ar, best_d, best_ma))
model_fit = model.fit()
model_fit.summary()

df_arima_resul_1 = pd.read_html(
    model_fit.summary().tables[1].as_html(), header=0)[0]
df_arima_resul_1.rename(columns=cols, inplace=True)
# Exportar a Latex
# r  = 'Tabla_py_resultado_ARIMA_1.tex'
# df_arima_resul_1.to_latex(r, index=False, float_format="{:0.3f}".format,
#                           encoding='latin1',
#                           column_format='ccccccc')


#### Mejor modelo sin contante
model = ARIMA(y_train.values, order=(best_ar, best_d, best_ma), trend='n')
model_fit = model.fit()
model_fit.summary()

df_arima_resul_2 = pd.read_html(
    model_fit.summary().tables[1].as_html(), header=0)[0]
df_arima_resul_2.rename(columns=cols, inplace=True)
# Exportar a Latex
# r  = 'Tabla_py_resultado_ARIMA_2.tex'
# df_arima_resul_2.to_latex(r, index=False, float_format="{:0.3f}".format,
#                           column_format='ccccccc')



#%% Simulacion
y_est = pd.DataFrame()
# Simlaciones
for t in y_test.index:
    print(t)
    model = ARIMA(y[:t].values, order=(best_ar, best_d, best_ma), trend='n')
    model_fit = model.fit()
    output = model_fit.forecast()
    y_hat = pd.DataFrame.from_dict({t: output[0]}, orient='index')
    y_est = pd.concat([y_est, y_hat])



r = '/Datos/pronostico_arima_{ar}_{d}_{ma}.xlsx'
r = r.format(ar=best_ar, d=best_d, ma=best_ma)
y_est.to_excel(r)

