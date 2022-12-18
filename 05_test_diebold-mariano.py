"""
Comparacion de Modelos
"""

from dm_test import dm_test
import pandas as pd

r = '/Datos/errores v2 (edit sin Efecto Pandemia).xlsx'
df_pronosticos = pd.read_excel(r, sheet_name='Pronosticos')
df_pronosticos.set_index('Fecha', inplace=True)

cols_types = {
    'y_true':float,
    'y_pred_1_1':float, 'y_pred_1_2':float, 'y_pred_1_3':float,
    'y_pred_2_1':float, 'y_pred_2_2':float, 'y_pred_2_3':float,
    'y_pred_3_1':float, 'y_pred_3_2':float, 'y_pred_3_3':float,
    'y_pred_4_1':float, 'y_pred_4_2':float, 'y_pred_4_3':float,
    'y_pred_5_1':float
    }
df_pronosticos = df_pronosticos.astype(cols_types)

homologacion_nombres = {
    '1_1': 'RSV 1',
    '1_2': 'RSV 2',
    '1_3': 'RSV 3',
    '2_1': 'ACP+RSV 1',
    '2_2': 'ACP+RSV 2',
    '2_3': 'ACP+RSV 3',
    '3_1': 'ACPK+RSV 1',
    '3_2': 'ACPK+RSV 2',
    '3_3': 'ACPK+RSV 3',
    '4_1': 'ACI+RSV 1',
    '4_2': 'ACI+RSV 2',
    '4_3': 'ACI+RSV 3',
    '5_1': 'ARIMA'
    }


#%% Calculo de prueba Diebold y Mariano
estadisticos = {}
p_valores = {}
real_lst = df_pronosticos['y_true'].to_list()

for i in homologacion_nombres.keys():
    estadisticos.update({i: {} })
    p_valores.update({i: {} })
    
    for j in homologacion_nombres.keys():
                
        if i == j:
            pass
        
        else:
        
            pred1_lst = df_pronosticos['y_pred_'+i].to_list()
            pred2_lst = df_pronosticos['y_pred_'+j].to_list()
            
            rt = dm_test(real_lst, pred1_lst, pred2_lst, h = 1, crit="MSE")
            dm_estadistico = rt[0]
            dm_p_valor = rt[1]
            
            estadisticos[i].update({j: round(dm_estadistico, 3)})
            p_valores[i].update({j: round(dm_p_valor, 2)})
            # p_valores[i].update({j: dm_p_valor})
        
        
        
df_estadisticos = pd.DataFrame.from_dict(estadisticos)
df_estadisticos.sort_index(ascending=True, inplace=True)
df_estadisticos.rename(columns=homologacion_nombres, inplace=True)
df_estadisticos.rename(index=homologacion_nombres, inplace=True)

df_p_valores = pd.DataFrame.from_dict(p_valores)
df_p_valores.sort_index(ascending=True, inplace=True)
df_p_valores.rename(columns=homologacion_nombres, inplace=True)
df_p_valores.rename(index=homologacion_nombres, inplace=True)
df_p_valores.fillna('-', inplace=True)



#%% Guardar en Excel
r = '/Datos/Diebold y Mariano - Estadisticos.xlsx'
df_estadisticos.to_excel(r, index=True)

r = '/Datos/Diebold y Mariano - P Valores.xlsx'
df_p_valores.to_excel(r, index=True)


#%% Guardar en Latex
ruta = 'Tabla_DM_p_valores.tex'
df_p_valores.to_latex(ruta, column_format='cccccccccccccc')




