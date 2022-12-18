"""
Construccion de variables
"""

import pandas as pd
import ta

#%% Importaccion
ruta = '/Datos/Serie USDCOP.xlsx'
df = pd.read_excel(ruta)
df.rename(columns = {
    'Fecha (dd/mm/aaaa)': 'FECHA',
    'Número de negociaciones': 'NUMERO NEGOCIACIONES',
    'Monto negociado (en millones de dólares estadounidenses)': 'VOLUMEN',
    'Tasa de cambio de apertura': 'APERTURA',
    'Tasa promedio ponderado': 'PROMEDIO',
    'Tasa de cambio de cierre': 'CIERRE',
    'Tasa de cambio máxima': 'MAXIMO',
    'Tasa de cambio mínima': 'MINIMO'
    }, inplace = True)


#%%% Calculo de Retornos

#### Retornos
df.sort_values('FECHA', ascending = True, inplace = True)
df.reset_index(inplace = True, drop = True)
df['RETORNO_T_0'] = df['PROMEDIO'].pct_change(1)
df['RETORNO_T_0'] = df['RETORNO_T_0'].mul(100)

#### Rezagos
lags = 5
for i in range(1, lags+1):
    df['RETORNO_T_' + str(i)] = df['RETORNO_T_0'].shift(i)
    

#%% Calculo de indicadores tecnicos

#### Momentum Indicators
# Kaufman’s Adaptive Moving Average (KAMA)
df['KAMA_T_0'] = df['CIERRE'] - ta.momentum.kama(df['CIERRE'])
# Relative Strength Index (RSI)
df['RSI_T_0'] = ta.momentum.rsi(df['CIERRE'])
# Stochastic Oscillator (George Lane) (STO)
df['STO_T_0'] = ta.momentum.stoch(high = df['MAXIMO'], low = df['MINIMO'], close= df['CIERRE'])
# Williams %R (WR)
df['WR_T_0'] = ta.momentum.williams_r(high = df['MAXIMO'], low = df['MINIMO'], close = df['CIERRE'])

### Trend Indicators
# Average Directional Movement Index (ADX)
df['ADX_T_0'] = ta.trend.adx(high = df['MAXIMO'], low = df['MINIMO'], close= df['CIERRE'])
# Commodity Channel Index (CCI)
df['CCI_T_0'] = ta.trend.cci(high = df['MAXIMO'], low = df['MINIMO'], close= df['CIERRE'])
# Exponential Moving Average (EMA)
df['EMA_T_0'] = df['CIERRE'] - ta.trend.ema_indicator(close = df['CIERRE'])
# Moving Average Convergence Divergence (MACD)
df['MACD_T_0'] = ta.trend.macd(close = df['CIERRE'])
# Simple Moving Average (SMA)
df['SMA_T_0'] = df['CIERRE'] - ta.trend.sma_indicator(close = df['CIERRE'])

#### Volatility Indicators
# Average True Range (ATR)
df['ATR_T_0'] = ta.volatility.average_true_range(high = df['MAXIMO'], low = df['MINIMO'], close= df['CIERRE'])

#### Volumne Indicators
# Chaikin Money Flow (CMF)
df['CMF_T_0'] = ta.volume.chaikin_money_flow(high = df['MAXIMO'], low = df['MINIMO'], close= df['CIERRE'], volume = df['VOLUMEN'])
# On-balance volume (OBV)
df['OBV_T_0'] = ta.volume.on_balance_volume(close = df['CIERRE'], volume = df['VOLUMEN'])


#%% Calculado rezagos de indicadores tecnicos
indicadores = ['KAMA', 'RSI', 'STO', 'WR', 'ADX', 'CCI',
               'EMA', 'MACD', 'SMA', 'ATR', 'CMF', 'OBV'] # Ind. a rezagar
lags_it = [1] # periodos que se rezaga cada indicador

for i in indicadores:
    for j in lags_it:
        name_lag = i + ' T_' + str(j)
        df[name_lag] = df[i + '_T_0'].shift(j)

del(i, j, indicadores, lags_it, name_lag)

#%% Exporta datos
ruta_save = '/Datos/Datos_arreglados.xlsx'
df.to_excel(ruta_save, index = False)
