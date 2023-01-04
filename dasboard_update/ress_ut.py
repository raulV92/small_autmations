# ya no quiero calcular estos 3 indicadores...

from pathlib import Path
import os

import numpy as np
import pandas as pd

import pretty_errors

def read_e2e(e2e_dir):
    ''' Funcion para leer los Audit Summary en la carpeta con nombre del argumento de funcion
    '''
    work_folder = e2e_dir

    all_files = os.listdir(work_folder)
    csv_files = filter(lambda x: x[-4:] == '.csv', all_files)
    csv_files = list(csv_files)
    # print(csv_files)

    all_csv = []
    e2e_cols = [
        'COUNTRY_ID', 'PERIOD_GROUP', 'PERIOD_ID', 'AUDTIOR_ID','ACTIVITY_LOCAL_NM',
        'RMS_RESOURCE_NM', 'STORE_STATUS', 'SMS_ID',  'TOTAL_ITEMS'
    ]
    for f, i in enumerate(csv_files):
        print('Leyendo archivo: ', i)

        individual_csv = pd.read_csv(Path(work_folder, i), sep='\t', encoding='utf_16_le',usecols=e2e_cols)

        ## NOTE
        # Caso Brasil:
        if 76 in individual_csv['COUNTRY_ID'].unique():
            # quita 'Antecipa Compras'
            individual_csv = individual_csv[individual_csv['ACTIVITY_LOCAL_NM']!='Antecipa Compras']

        ############
        all_csv.append(individual_csv)
        # print('Periodos: ', all_csv[f]['PERIOD_ID'].unique())

    try:
        all_csv = pd.concat(all_csv, ignore_index=True)
    except:
        print(
            'Por favor verifica que la carpeta "updated_E2E_input" no este vacia'
        )

    # Termina de leer local

    countries = {
        32: 'Argentina',
        858: 'Uruguay',
        484: 'Mexico',
        76: 'Brasil',
        68: 'Bolivia',
        152: 'Chile',
        218: 'Ecuador',
        170: 'Colombia',
        862: 'Venezuela',
        222: 'El Salvador',
        320: 'Guatemala',
        340: 'Honduras',
        558: 'Nicaragua',
        591: 'Panama',
        600: 'Paraguay',
        604: 'Peru',
        630: 'Puerto Rico',
        214: 'Republica Dominicana',
        188: 'Costa Rica'
    }


    df = all_csv[e2e_cols].copy()

    # normaliza tipos de sms id y codigo de auditor:
    tipos = {'SMS_ID': str, 'AUDTIOR_ID': np.int32}
    df = df.astype(tipos).drop_duplicates()

    # cambia nombre de columnas para join, limpia sms id

    df.rename(columns={
        'PERIOD_GROUP': 'Frequency',
        'AUDTIOR_ID': 'Visit CDAR ID',
        # 'SMS_ID': 'SMS ID'
    },
              inplace=True)

    # elimina contratacion, elimina parciales, codigo de pais a nombre:
    df = df[df['SMS_ID'].apply(lambda x: len(x)) == 10]
    # elminar numeros a 10 digitos 'negativos'
    df = df[~df['SMS_ID'].apply(lambda x: x.startswith('-'))]

    '''
    df['VISIT_DATE'].where(df['STORE_STATUS'] != 'PARTIAL',
                           other=' ',
                           inplace=True)'''

    df['COUNTRY_ID'].replace(to_replace=countries, inplace=True)
    '''
    df['VISIT_DATE'] = pd.to_datetime(df['VISIT_DATE'].str.replace(' ', ''),
                                      format='%m/%d/%Y',
                                      errors='coerce')'''

    # df.sort_values(by=['SMS ID', 'VISIT_DATE'])
    df = df[~df.duplicated(subset=['SMS_ID', 'Frequency'], keep='last')]

    # df.to_csv(Path('ress_utilization','concat_E2E.csv', index=False)
    # print("2) Exportado: 'concat_E2E.csv'")

    return df

def stores_per_auditor(pais, df):
    # print(pais)
    # print(df)
    stores_by_auditor=df.pivot_table(index='RMS_RESOURCE_NM',values='SMS_ID',aggfunc=len)
    stores_by_auditor.sort_values(by='SMS_ID',inplace=True)
    stores_by_auditor['accum']=stores_by_auditor['SMS_ID'].cumsum()
    stores_by_auditor['perc']=stores_by_auditor['accum']/stores_by_auditor['SMS_ID'].sum()
    filter_stores_by_auditor=stores_by_auditor[(0.2<stores_by_auditor['perc']) & (stores_by_auditor['perc']<0.8)]
    stores_by_auditor.to_csv(Path('ress_utilization',pais+'_pareto_stores.csv'))

    res_line=(pais,filter_stores_by_auditor['SMS_ID'].sum(),len(filter_stores_by_auditor),filter_stores_by_auditor['SMS_ID'].mean())
    return res_line
    # breakpoint()

def lines_per_auditor(pais,df):
    # df.to_csv(Path('ress_utilization','sanity_lineas.csv')
    lines_by_aud=df.pivot_table(index='RMS_RESOURCE_NM',values='TOTAL_ITEMS',aggfunc=sum)
    lines_by_aud.sort_values(by='TOTAL_ITEMS',inplace=True)
    lines_by_aud['accum']=lines_by_aud['TOTAL_ITEMS'].cumsum()
    lines_by_aud['perc'] = lines_by_aud['accum']/lines_by_aud['TOTAL_ITEMS'].sum()
    filter_lines_by_aud=lines_by_aud[(0.2<lines_by_aud['perc']) & (lines_by_aud['perc']<0.8)]
    lines_by_aud.to_csv(Path('ress_utilization',pais+'_pareto_lines.csv'))

    res_line= (pais,filter_lines_by_aud['TOTAL_ITEMS'].sum(),len(filter_lines_by_aud),filter_lines_by_aud['TOTAL_ITEMS'].mean())
    return res_line
    # breakpoint()


if __name__ == '__main__':
    test_dir= Path('updated_E2E_input','cierre_dic')

    df = read_e2e(test_dir)
    # breakpoint()
    paises = df.groupby(by='COUNTRY_ID')
    tiendas_por_auditor=[]
    lineas_por_auditor = []
    for country, auditSummary in paises:

        tabla_pais=auditSummary[['RMS_RESOURCE_NM','SMS_ID']].drop_duplicates()
        tiendas_por_auditor.append(stores_per_auditor(country,tabla_pais))
        # breakpoint()
        pais_con_items=auditSummary[['RMS_RESOURCE_NM','SMS_ID','TOTAL_ITEMS']].drop_duplicates()
        # breakpoint()
        lineas_por_auditor.append(lines_per_auditor(country,pais_con_items))

    tiendas_por_auditor = pd.DataFrame(tiendas_por_auditor,columns=['Pais','Tiendas Totales','Auditores','Promedio'])
    lineas_por_auditor = pd.DataFrame(lineas_por_auditor,columns=['Pais','Items Totales','Auditores','Promedio'])

    tiendas_por_auditor.to_csv(Path('ress_utilization','KPI_tiendas_por_auditor.csv'),index=False)
    lineas_por_auditor.to_csv(Path('ress_utilization','KPI_lineas_por_auditor.csv'),index=False)
    #breakpoint()
    # stores_per_auditor(df['','AUDTIOR_ID','SMS_ID'].drop_duplicates())


